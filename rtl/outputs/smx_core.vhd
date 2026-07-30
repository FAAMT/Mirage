-- =============================================================================
--  File      : smx_core.vhd  (DRAFT — compile-unverified: no simulator in run)
--  Purpose   : Provisional SMX wrapper: taylor_exp (reference) + smx_cache
--              (outputs/) + smx_v_calc (outputs/), score row in -> attention out.
--  Notes     : The exp-input scale fix point (W3/W5: score Q9 + 1/sqrt(64) via
--              <<6, plus range reduction) belongs INSIDE taylor_exp upstream;
--              the SCALE_SHIFT generic here marks the injection point.
--              Compile-unverified draft; port maps follow the reference
--              taylor_exp entity and the outputs/ drafts.
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity smx_core is
    generic
    (
        NUM_HEADS : positive := 8;
        MAX_TOK   : positive := 16;
        HEAD_DIM  : positive := 64;
        EXP_W     : positive := 32;
        SUM_W     : positive := 48
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        -- Score row input: one query row = NUM_HEADS*TOK scores (from qk_score_bram)
        score_row_valid : in std_logic;
        score_row       : in score_io_type;

        -- V stream (from QKV BRAM), one element per cycle
        v_valid : in std_logic;
        v_key   : in integer range 0 to MAX_TOK - 1;
        v_dim   : in integer range 0 to HEAD_DIM - 1;
        v_data  : in signed(7 downto 0);

        -- Attention output stream
        attn_valid : out std_logic;
        attn_dim   : out integer range 0 to HEAD_DIM - 1;
        attn_data  : out signed(EXP_W - 1 downto 0)
    );
end entity;

architecture structural of smx_core is

    -- taylor_exp <-> smx_cache / smx_v_calc
    signal exp_valid : std_logic := '0';
    signal exp_head  : integer range 0 to NUM_HEADS - 1 := 0;
    signal exp_last  : std_logic := '0';
    signal exp_data  : signed(EXP_W - 1 downto 0) := (others => '0');
    signal tok_q     : integer range 0 to MAX_TOK - 1 := 0;

    -- smx_cache <-> smx_v_calc
    signal row_done : std_logic;
    signal rd_head  : integer range 0 to NUM_HEADS - 1 := 0;
    signal rd_sum   : signed(SUM_W - 1 downto 0);

    signal vcalc_done : std_logic;

begin

    ---------------------------------------------------------------------------
    -- Exp stage: reference taylor_exp (needs the W3/W5 fixes upstream:
    -- Q9 per-term shifts, <<6 exp-input scale, range reduction, done driver).
    -- Its streaming per-(head,key) outputs feed exp_* here; glue TBD upstream.
    ---------------------------------------------------------------------------
    -- taylor_exp_inst : entity work.taylor_exp
    --     port map ( ... score_row ..., exp_stream => ... );

    ---------------------------------------------------------------------------
    -- Partial-sum cache (outputs/ draft)
    ---------------------------------------------------------------------------
    cache_inst : entity work.smx_cache
        generic map (NUM_HEADS => NUM_HEADS, MAX_TOK => MAX_TOK,
                     EXP_W => EXP_W, SUM_W => SUM_W)
        port map (
            clk       => clk,
            rst       => rst,
            exp_valid => exp_valid,
            exp_head  => exp_head,
            exp_last  => exp_last,
            exp_data  => exp_data,
            tok_q     => tok_q,
            row_done  => row_done,
            rd_head   => rd_head,
            rd_tok    => tok_q,
            rd_sum    => rd_sum
        );

    ---------------------------------------------------------------------------
    -- Normalize + weighted V sum (outputs/ draft)
    ---------------------------------------------------------------------------
    vcalc_inst : entity work.smx_v_calc
        generic map (MAX_TOK => MAX_TOK, HEAD_DIM => HEAD_DIM,
                     EXP_W => EXP_W, SUM_W => SUM_W)
        port map (
            clk           => clk,
            rst           => rst,
            exp_valid     => exp_valid,
            exp_last      => exp_last,
            exp_data      => exp_data,
            row_sum       => rd_sum,
            row_sum_valid => row_done,
            v_valid       => v_valid,
            v_key         => v_key,
            v_dim         => v_dim,
            v_data        => v_data,
            attn_valid    => attn_valid,
            attn_dim      => attn_dim,
            attn_data     => attn_data,
            done          => vcalc_done
        );

    done <= vcalc_done;

end architecture;
