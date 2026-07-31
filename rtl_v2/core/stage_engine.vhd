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
-- ADDRESS GENERATION
--
--   xpose = '0'  (xV)   : addr = base + row*stride + lane
--   xpose = '1'  (QK^T) : addr = base + lane*stride + row
--
-- Both are one address per cycle with no multiplier: the two products are
-- kept as running accumulators, advanced when the corresponding counter
-- ticks, for the same timing reason as in gemm_sequencer.
--
-- `stride` is the distance in activation memory between consecutive
-- TOKENS, which is TOK_STRIDE = MODEL_DIM, and has nothing to do with how
-- many lanes or rows this particular pass covers.
--
-- FOUR BUGS THIS FILE USED TO HAVE, all found by sim/stage_engine_asis_tb
-- (45 of 160 staged bytes wrong on the two geometries layer_ctrl issues):
--
--   1. STRIDE. The address used `rows` when transposing and `lanes` when
--      not -- the *shape of the pass* rather than the *layout of memory*.
--      For QK^T that made the token stride 64 instead of 512, so every
--      lane but lane 0 read the wrong token. There is now an explicit
--      `stride` port and the caller supplies TOK_STRIDE.
--
--   2. BANK INDEX. `bank_of` was a free-running counter that was never
--      reset per row, so with lanes = 4 (one bank word per row) row r
--      landed in bank r. A row's banks must start at 0 every time, so the
--      bank index is now DERIVED from the lane counter and cannot drift.
--
--   3. ROW INDEX SAMPLED A CYCLE LATE. The packer runs one cycle behind
--      the address generator but read `row_cnt` live, so the last bank
--      word of every row was written to row+1. The row now travels down
--      the pipeline WITH the data it belongs to.
--
--   4. SHORT ROWS LEFT STALE DATA. A pass with lanes < N_LANES wrote only
--      the first lanes/4 banks and left the rest holding whatever the
--      previous op staged, so unused lanes accumulated garbage into real
--      output columns. Every row is now written in full, N_LANES wide,
--      with lanes >= `lanes` forced to zero. A zero weight contributes
--      nothing to a dot product, which is the only safe filler.
--
-- Cost of (4): a pass is always rows x N_LANES cycles rather than
-- rows x lanes. For QK^T that is 64 x 32 = 2048 cycles against the 1024
-- of the matmul it feeds. The factor of two on the attention matmuls was
-- always the accepted trade here; this makes it exact rather than
-- data-dependent.
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
        base   : in std_logic_vector(ACT_AW - 1 downto 0);
        rows   : in std_logic_vector(6 downto 0);   -- reduction length
        lanes  : in std_logic_vector(6 downto 0);   -- lanes carrying data
        stride : in std_logic_vector(ACT_AW - 1 downto 0);  -- token stride
        xpose  : in std_logic;
        -- Window row base. xV needs HEAD_DIM = 64 lanes but a window row
        -- is only N_LANES = 32 wide, so it is staged as two passes whose
        -- windows sit at rows 0.. and rows.. respectively -- matching the
        -- w_tile = ct * k_len stepping gemm_sequencer already does.
        row_ofs : in std_logic_vector(BANK_AW - 1 downto 0);

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

    type state_t is (S_IDLE, S_READ, S_DRAIN, S_FIN);
    signal state : state_t := S_IDLE;

    signal row_cnt  : unsigned(6 downto 0) := (others => '0');
    signal lane_cnt : unsigned(6 downto 0) := (others => '0');

    -- running products, so no multiplier sits in the address path
    signal row_acc  : unsigned(ACT_AW - 1 downto 0) := (others => '0');
    signal lane_acc : unsigned(ACT_AW - 1 downto 0) := (others => '0');

    -- byte packer: BANK_BYTES bytes make one bank word
    signal pack : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');

    -- Read pipeline, TWO stages deep, because the address is registered
    -- and the memory then takes a cycle:
    --
    --   edge N    address + act_re issued          (stage 1 tags set)
    --   edge N+1  memory returns the byte          (stage 1 -> stage 2)
    --   edge N+2  packer consumes byte + stage-2 tags
    --
    -- Everything the packer needs about a byte -- which lane it is, which
    -- window row it belongs to, whether it is real or zero padding --
    -- travels down these stages WITH it, so nothing is re-read from a
    -- counter that has since moved on.
    signal rd_v,    rd2_v    : std_logic := '0';
    signal rd_zero, rd2_zero : std_logic := '0';
    signal rd_lane, rd2_lane : unsigned(6 downto 0) := (others => '0');
    signal rd_row,  rd2_row  : unsigned(6 downto 0) := (others => '0');

    signal drain : unsigned(1 downto 0) := (others => '0');

    signal busy_i : std_logic := '0';
    signal fin_i  : std_logic := '0';

    -- The byte actually packed: the activation, or zero for a lane this
    -- pass does not cover.
    signal byte_in : std_logic_vector(ACT_W - 1 downto 0);

begin

    busy <= busy_i;
    fin  <= fin_i;

    byte_in <= (others => '0') when rd2_zero = '1' else std_logic_vector(act_data);

    proc : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                state    <= S_IDLE;
                row_cnt  <= (others => '0');
                lane_cnt <= (others => '0');
                row_acc  <= (others => '0');
                lane_acc <= (others => '0');
                rd_v     <= '0';
                rd_zero  <= '0';
                rd_lane  <= (others => '0');
                rd_row   <= (others => '0');
                rd2_v    <= '0';
                rd2_zero <= '0';
                rd2_lane <= (others => '0');
                rd2_row  <= (others => '0');
                drain    <= (others => '0');
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
                            drain    <= (others => '0');
                            state    <= S_READ;
                        end if;

                    when S_READ =>
                        act_re <= '1';
                        -- The address is registered HERE, in the same cycle
                        -- as act_re, rather than assigned concurrently from
                        -- the live counters.
                        --
                        -- It used to be a concurrent statement, which put it
                        -- one cycle AHEAD of the registered act_re: by the
                        -- time act_re went high the counters had already
                        -- advanced, so the memory returned lane 1's byte
                        -- while the pipeline labelled it lane 0. From lane 1
                        -- on the two stayed aligned and every later byte was
                        -- right, so the whole fault was exactly ONE wrong
                        -- byte per pass -- the first. sim/stage_engine_tb
                        -- caught it as 4 errors in 2,368 checks, which is
                        -- precisely the kind of defect an end-to-end test
                        -- would have averaged away.
                        --
                        -- gemm_sequencer registers act_re and act_raddr
                        -- together and always did; this makes the two
                        -- modules agree.
                        if xpose = '1' then
                            act_raddr <= std_logic_vector(unsigned(base) + lane_acc
                                                          + resize(row_cnt, ACT_AW));
                        else
                            act_raddr <= std_logic_vector(unsigned(base) + row_acc
                                                          + resize(lane_cnt, ACT_AW));
                        end if;
                        -- Everything the packer will need about THIS byte,
                        -- captured now rather than re-read a cycle later.
                        rd_v    <= '1';
                        rd_lane <= lane_cnt;
                        rd_row  <= row_cnt;
                        if lane_cnt < unsigned(lanes) then
                            rd_zero <= '0';
                        else
                            rd_zero <= '1';
                        end if;

                        -- Always walk the full N_LANES: short rows are
                        -- zero-filled, never left stale.
                        if lane_cnt = N_LANES - 1 then
                            lane_cnt <= (others => '0');
                            lane_acc <= (others => '0');
                            if row_cnt = unsigned(rows) - 1 then
                                state <= S_DRAIN;
                            else
                                row_cnt <= row_cnt + 1;
                                row_acc <= row_acc + unsigned(stride);
                            end if;
                        else
                            lane_cnt <= lane_cnt + 1;
                            lane_acc <= lane_acc + unsigned(stride);
                        end if;

                    when S_DRAIN =>
                        -- Two cycles to walk the last byte through the read
                        -- pipeline, plus one for the packer's write to
                        -- commit, before `fin` releases layer_ctrl. Leaving
                        -- early would drop the final bank word of the last
                        -- row -- a corrupted last output column that only
                        -- shows up as slightly wrong numbers.
                        if drain = 2 then
                            state <= S_FIN;
                        else
                            drain <= drain + 1;
                        end if;

                    when S_FIN =>
                        fin_i  <= '1';
                        busy_i <= '0';
                        state  <= S_IDLE;

                end case;

                -- Stage 1 -> stage 2. The byte the memory is returning this
                -- cycle was tagged one cycle ago, so the tags move down to
                -- meet it.
                rd2_v    <= rd_v;
                rd2_zero <= rd_zero;
                rd2_lane <= rd_lane;
                rd2_row  <= rd_row;

                -- Packer, two cycles behind the address issue so it sees the
                -- data that address returned. Bank and row come from the
                -- pipeline registers, never from the live counters.
                if rd2_v = '1' then
                    pack <= byte_in & pack(BANK_W - 1 downto ACT_W);
                    if rd2_lane(1 downto 0) = BANK_BYTES - 1 then
                        w_we   <= '1';
                        w_data <= byte_in & pack(BANK_W - 1 downto ACT_W);
                        w_row  <= std_logic_vector(unsigned(row_ofs)
                                                   + resize(rd2_row, BANK_AW));
                        -- Bank is DERIVED from the lane, so a row's banks
                        -- always start at 0. The free-running counter this
                        -- replaced put row r's single bank word in bank r
                        -- whenever a row was only four lanes wide.
                        w_bank <= std_logic_vector(rd2_lane(4 downto 2));
                    end if;
                end if;

            end if;
        end if;
    end process;

end architecture;
