-- =============================================================================
--  File      : smx_cache.vhd  (DRAFT — compile-unverified: no simulator in run)
--  Purpose   : Softmax exp partial-sum cache. Accumulates the streaming
--              taylor_exp outputs (Q-format e^{z-max} values) into per-head
--              partial sums, keyed by (head, query token). Eviction: none —
--              sums stream out to the divide/normalize stage once per row.
--  Notes     : Standalone (no types_pkg dependency) so token count can scale
--              past the reference design's TOK_CONTEXT=4 (acceptance test is
--              8 heads x 16 tokens). Integer accumulation is exact: the cache
--              itself introduces 0 LSB drift; all error budget lives upstream
--              in the exp core (see reports/w3-exp-error-histogram.md).
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity smx_cache is
    generic
    (
        NUM_HEADS : positive := 8;   -- attention heads
        MAX_TOK   : positive := 16;  -- max query tokens (context) supported
        EXP_W     : positive := 32;  -- width of one Q-format exp input
        SUM_W     : positive := 48   -- width of accumulated partial sum
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- Streaming exp input: one (head, key) exp value per valid cycle,
        -- belonging to the query row currently selected by tok_q.
        exp_valid : in std_logic;
        exp_head  : in integer range 0 to NUM_HEADS - 1;
        exp_last  : in std_logic;                    -- '1' on the last key of this head's row
        exp_data  : in signed(EXP_W - 1 downto 0);

        tok_q     : in integer range 0 to MAX_TOK - 1;

        -- Pulses for one cycle when every head's partial sum for tok_q
        -- has been stored (row ready for the divide stage).
        row_done  : out std_logic;

        -- Combinational cache readout (keyed by head/token)
        rd_head : in integer range 0 to NUM_HEADS - 1;
        rd_tok  : in integer range 0 to MAX_TOK - 1;
        rd_sum  : out signed(SUM_W - 1 downto 0)
    );
end entity;

architecture rtl of smx_cache is

    type sum_array is array (0 to NUM_HEADS - 1) of signed(SUM_W - 1 downto 0);
    type cache_type is array (0 to NUM_HEADS - 1, 0 to MAX_TOK - 1) of signed(SUM_W - 1 downto 0);

    signal cache     : cache_type := (others => (others => (others => '0')));
    signal accum     : sum_array  := (others => (others => '0'));
    signal head_done : std_logic_vector(0 to NUM_HEADS - 1) := (others => '0');

begin

    -- Combinational readout
    rd_sum <= cache(rd_head, rd_tok);

    process (clk, rst)
    begin
        if rst = '1' then
            accum     <= (others => (others => '0'));
            head_done <= (others => '0');
            row_done  <= '0';

        elsif rising_edge(clk) then
            row_done <= '0';

            if exp_valid = '1' then
                if exp_last = '1' then
                    -- Close this head's partial sum for the current row
                    cache(exp_head, tok_q) <= accum(exp_head) + resize(exp_data, SUM_W);
                    accum(exp_head)        <= (others => '0');
                    head_done(exp_head)    <= '1';
                else
                    accum(exp_head) <= accum(exp_head) + resize(exp_data, SUM_W);
                end if;
            end if;

            -- Row completion: all heads stored (including any store this cycle)
            if (and head_done) = '1' then
                row_done  <= '1';
                head_done <= (others => '0');
            end if;
        end if;
    end process;

end architecture;
