-- =====================================================================
-- stage_engine.vhd — copies an activation operand into the weight window
--
-- Six of the encoder's eight matmuls take a streamed weight matrix as the
-- per-lane operand, which weight_window serves directly. The other two --
-- QK^T and xV -- take another ACTIVATION there. This engine bridges that,
-- reading activation memory a byte per cycle, packing BANK_BYTES of them
-- into a bank word, and writing it through weight_window's existing write
-- port.
--
-- The alternative was a second wide read port on activation memory. That
-- memory is 8 bits wide, so it could not supply N_LANES bytes per cycle
-- however many ports it had; widening it would have meant banking the
-- activations too, for two matmuls out of eight. Staging costs cycles
-- instead of area, which is the right way round on a device whose LUTs v1
-- had already blown through by 26%.
--
-- ADDRESS GENERATION -- the part that carries the transpose
--
--   xpose = '0'  (xV)   : row-major, addr = base + row*lanes  + lane
--   xpose = '1'  (QK^T) : transposed, addr = base + lane*rows + row
--
-- Both are one address per cycle with no multiplier: the two products are
-- kept as running accumulators, advanced when the corresponding counter
-- ticks, for the same timing reason as in gemm_sequencer.
--
-- Writes land in the buffer NOT selected for compute, reusing
-- weight_window's double-buffer machinery unchanged.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity stage_engine is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- control
        go    : in  std_logic;
        busy  : out std_logic;
        fin   : out std_logic;

        -- geometry
        base  : in std_logic_vector(ACT_AW - 1 downto 0);
        rows  : in std_logic_vector(6 downto 0);   -- reduction length
        lanes : in std_logic_vector(6 downto 0);   -- lanes to fill
        xpose : in std_logic;

        -- activation memory read (muxed against the sequencer upstream)
        act_re    : out std_logic;
        act_raddr : out std_logic_vector(ACT_AW - 1 downto 0);
        act_data  : in  act_t;

        -- weight window write
        w_we    : out std_logic;
        w_row   : out std_logic_vector(BANK_AW - 1 downto 0);
        w_bank  : out std_logic_vector(2 downto 0);
        w_data  : out std_logic_vector(BANK_W - 1 downto 0)
    );
end entity;

architecture rtl of stage_engine is

    type state_t is (S_IDLE, S_READ, S_FIN);
    signal state : state_t := S_IDLE;

    signal row_cnt  : unsigned(6 downto 0) := (others => '0');
    signal lane_cnt : unsigned(6 downto 0) := (others => '0');

    -- running products, so no multiplier sits in the address path
    signal row_acc  : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal lane_acc : unsigned(ACT_AW - 1 downto 0) := (others => '0');

    -- byte packer: BANK_BYTES bytes make one bank word
    signal pack     : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');
    signal pack_cnt : unsigned(1 downto 0) := (others => '0');

    -- act memory read latency is 1 cycle, so the packer runs one behind
    signal rd_v    : std_logic := '0';
    signal bank_of : unsigned(2 downto 0) := (others => '0');

    signal busy_i : std_logic := '0';
    signal fin_i  : std_logic := '0';

begin

    busy <= busy_i;
    fin  <= fin_i;

    -- addr = base + (xpose ? lane*rows + row : row*lanes + lane)
    act_raddr <= std_logic_vector(unsigned(base) + lane_acc + resize(row_cnt, ACT_AW))
                 when xpose = '1' else
                 std_logic_vector(unsigned(base) + row_acc + resize(lane_cnt, ACT_AW));

    proc : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state    <= S_IDLE;
                row_cnt  <= (others => '0');
                lane_cnt <= (others => '0');
                row_acc  <= (others => '0');
                lane_acc <= (others => '0');
                pack_cnt <= (others => '0');
                bank_of  <= (others => '0');
                rd_v     <= '0';
                busy_i   <= '0';
                fin_i    <= '0';
                act_re   <= '0';
                w_we     <= '0';
            else
                act_re <= '0';
                w_we   <= '0';
                fin_i  <= '0';
                rd_v   <= '0';

                case state is

                    when S_IDLE =>
                        busy_i <= '0';
                        if go = '1' then
                            busy_i   <= '1';
                            row_cnt  <= (others => '0');
                            lane_cnt <= (others => '0');
                            row_acc  <= (others => '0');
                            lane_acc <= (others => '0');
                            pack_cnt <= (others => '0');
                            bank_of  <= (others => '0');
                            state    <= S_READ;
                        end if;

                    when S_READ =>
                        act_re <= '1';
                        rd_v   <= '1';

                        -- advance lane, then row
                        if lane_cnt = unsigned(lanes) - 1 then
                            lane_cnt <= (others => '0');
                            lane_acc <= (others => '0');
                            if row_cnt = unsigned(rows) - 1 then
                                state <= S_FIN;
                            else
                                row_cnt <= row_cnt + 1;
                                row_acc <= row_acc + resize(unsigned(lanes), ACT_AW);
                            end if;
                        else
                            lane_cnt <= lane_cnt + 1;
                            lane_acc <= lane_acc + resize(unsigned(rows), ACT_AW);
                        end if;

                    when S_FIN =>
                        -- let the last read and its pack write drain
                        fin_i  <= '1';
                        busy_i <= '0';
                        state  <= S_IDLE;

                end case;

                -- Packer, one cycle behind the address generator so it sees
                -- the data that address returned.
                if rd_v = '1' then
                    pack <= std_logic_vector(act_data) & pack(BANK_W - 1 downto ACT_W);
                    if pack_cnt = BANK_BYTES - 1 then
                        pack_cnt <= (others => '0');
                        w_we     <= '1';
                        w_data   <= std_logic_vector(act_data)
                                    & pack(BANK_W - 1 downto ACT_W);
                        w_row    <= std_logic_vector(resize(row_cnt, BANK_AW));
                        w_bank   <= std_logic_vector(bank_of);
                        if bank_of = W_BANKS - 1 then
                            bank_of <= (others => '0');
                        else
                            bank_of <= bank_of + 1;
                        end if;
                    else
                        pack_cnt <= pack_cnt + 1;
                    end if;
                end if;

            end if;
        end if;
    end process;

end architecture;
