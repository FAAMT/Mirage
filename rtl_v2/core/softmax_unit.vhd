-- =====================================================================
-- softmax_unit.vhd — streaming, numerically-stable softmax front end
--
-- Consumes a stream of scores for one (head, query) row and produces a
-- stream of int8 exp values plus their sum. It does NOT produce
-- probabilities and contains no divider -- see recip_rom for why.
--
-- Two passes over the row, replayed by gemm_sequencer:
--
--   PASS_MAX : watch the scores, keep the maximum. Standard numerically
--              stable softmax: subtracting the max keeps every exp
--              argument <= 0, which is also exp_rom's documented domain.
--   PASS_EXP : z = score - max, look up exp(z), requantise to int8,
--              emit it, and accumulate sum8.
--
-- Replaying the row costs MAX_TOK extra cycles per (head, query) -- 16
-- against the ~512-cycle matmuls around it. The alternative, buffering
-- the row on chip, would cost a memory and a write port to save nothing
-- that matters.
--
-- v1's equivalent path was three modules (taylor_exp, smx_cache,
-- smx_v_calc) that between them measured 4,041 LUT and 10,400 FF, none of
-- it in BRAM, and taylor_exp was optimised away entirely because its
-- outputs were undriven. This unit is a datapath with no internal
-- addressing and every output driven, so there is nothing to prune.
--
-- Latency from score in to e8 out is EXP_LAT cycles (clamp, ROM, requant).
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity softmax_unit is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- Row framing. `row_start` clears the accumulators for a new
        -- (head, query); `pass_exp` selects which pass is running.
        row_start : in std_logic;
        pass_exp  : in std_logic;   -- '0' = PASS_MAX, '1' = PASS_EXP

        -- score stream in
        s_valid : in std_logic;
        s_data  : in score_t;

        -- int8 exp stream out (PASS_EXP only)
        e_valid : out std_logic;
        e_data  : out act_t;

        -- running sum of the emitted e8 values, valid once the pass ends
        sum8 : out std_logic_vector(SUM8_W - 1 downto 0)
    );
end entity;

architecture rtl of softmax_unit is

    -- exp_rom interface
    signal rom_re   : std_logic := '0';
    signal rom_addr : std_logic_vector(EXP_ROM_AW - 1 downto 0) := (others => '0');
    signal rom_dout : std_logic_vector(EXP_ROM_DW - 1 downto 0);

    signal max_q  : score_t := (others => '0');
    signal max_valid : std_logic := '0';

    -- Valid pipeline: address issue -> ROM read -> requantise.
    --
    -- THREE stages, not two. exp_rom is a SYNCHRONOUS read, so the data
    -- for the address issued alongside v1 does not appear until the cycle
    -- after v1 is high. The requantiser used to fire on v1 itself and
    -- therefore consumed the PREVIOUS score's exp value: the whole row
    -- came out shifted one column left.
    --
    -- That fault is close to invisible from the outside. The row still
    -- looks like a softmax, the values are still in range, and a row of
    -- equal scores -- the obvious thing to try first -- is invariant under
    -- the shift and passes. sim/softmax_unit_tb catches it by driving a
    -- strictly increasing row and comparing element by element.
    signal v1, v1_d, v2 : std_logic := '0';

    signal e_q    : act_t := (others => '0');
    signal sum_q  : unsigned(SUM8_W - 1 downto 0) := (others => '0');

begin

    e_valid <= v2;
    e_data  <= e_q;
    sum8    <= std_logic_vector(sum_q);

    -- MAG_MAX must be exactly the old clamp floor, or the ROM domain and
    -- the index range stop agreeing.
    assert MAG_MAX = -EXP_Z_MIN - 1
        report "MAG_MAX (" & integer'image(MAG_MAX) & ") /= -EXP_Z_MIN-1 ("
             & integer'image(-EXP_Z_MIN - 1) & ")"
        severity failure;

    exp_tbl : entity work.exp_rom
        port map (clk => clk, re => rom_re, addr => rom_addr, dout => rom_dout);

    ---------------------------------------------------------------------
    -- Stage 1: max tracking (PASS_MAX) and z -> ROM address (PASS_EXP)
    ---------------------------------------------------------------------
    front : process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                max_q     <= (others => '0');
                max_valid <= '0';
                rom_re    <= '0';
                v1        <= '0';
            else
                rom_re <= '0';
                v1     <= '0';

                if row_start = '1' then
                    -- Seed the max with the most negative score possible so
                    -- the first real sample always replaces it.
                    max_q     <= to_signed(-(2 ** (SCORE_W - 1)), SCORE_W);
                    max_valid <= '0';
                end if;

                if s_valid = '1' then
                    if pass_exp = '0' then
                        -- PASS_MAX
                        if max_valid = '0' or s_data > max_q then
                            max_q     <= s_data;
                            max_valid <= '1';
                        end if;
                    else
                        -- PASS_EXP. This was open-coded here with `integer`
                        -- variables, so the subtract, both clamps, the
                        -- negate and the divide all ran at 32 bits: run 8's
                        -- worst path was 17 CARRY4 levels from the score
                        -- BRAM to rom_addr (-6.226 ns, 62% logic). It is now
                        -- exp_index -- signed at the natural width, one
                        -- clamp instead of three, and a shift instead of a
                        -- division -- and is tested in exp_idx_tb.
                        rom_addr <= std_logic_vector(exp_index(max_q, s_data));
                        rom_re   <= '1';
                        v1       <= '1';
                    end if;
                end if;
            end if;
        end if;
    end process;

    ---------------------------------------------------------------------
    -- Stage 2: ROM data -> int8, accumulate sum8
    --
    -- exp comes back in Q(EXP_ROM_FRAC) where 1.0 = 2^16. Requantising to
    -- int8 is a multiply by INT8_MAX and a shift; INT8_MAX = 127 is one
    -- less than a power of two, so (x << 7) - x avoids a multiplier here
    -- and keeps this stage in fabric rather than competing for a DSP.
    ---------------------------------------------------------------------
    back : process (clk)
        variable scaled : unsigned(EXP_ROM_DW + 8 downto 0);
        variable e8     : integer;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                v1_d  <= '0';
                v2    <= '0';
                e_q   <= (others => '0');
                sum_q <= (others => '0');
            else
                -- v1 marks the cycle the ROM ADDRESS is presented; v1_d
                -- marks the cycle its DATA is available. The requantiser
                -- must run on the latter.
                v1_d <= v1;
                v2   <= v1_d;

                if row_start = '1' then
                    sum_q <= (others => '0');
                end if;

                if v1_d = '1' then
                    scaled := (resize(unsigned(rom_dout), scaled'length) sll 7)
                              - resize(unsigned(rom_dout), scaled'length);
                    e8 := to_integer(shift_right(scaled, EXP_ROM_FRAC));
                    if e8 > INT8_MAX then
                        e8 := INT8_MAX;
                    end if;
                    e_q   <= to_signed(e8, ACT_W);
                    sum_q <= sum_q + to_unsigned(e8, SUM8_W);
                end if;
            end if;
        end if;
    end process;

end architecture;
