-- =====================================================================
-- mirage_sizing_container.vhd
--
-- A SIZING HARNESS, NOT AN ARCHITECTURE.
--
-- Purpose: get one honest device total out of Vivado without a block
-- diagram and without inventing a datapath that does not exist yet.
--
-- It instantiates each ROOT module exactly once. "Root" means: not
-- already instantiated inside another module in this package.
--   qkv_proj_bram and quantizer live inside qkv_score_calc.
--   smx_cache and smx_v_calc live inside smx_core.
-- Synthesising those separately AND inside their parents double-counts
-- them, which is the main inaccuracy in a per-module-only run.
--
-- Every signal is either (a) wired to its real counterpart where the
-- source already defines an unambiguous, type-exact pairing, or
-- (b) brought straight out to a container port. Nothing is invented:
-- no arbiter, no width adapter, no fabricated control. Ports left
-- exposed are exposed precisely because the design does not yet say
-- what they connect to -- see KNOWN_ISSUES.md sections 4-6.
--
-- Synthesise with -mode out_of_context -flatten_hierarchy none so that
-- nothing is pruned at the boundary and report_utilization -hierarchical
-- gives a per-block breakdown and a true total from one run.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.types_pkg.all;

entity mirage_sizing_container is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        ---------------------------------------------------------------
        -- gemm_array (shared MAC tile; input side driven by
        -- qkv_score_calc internally -- the one complete MAC pairing)
        ---------------------------------------------------------------
        ga_output_d : out std_logic_vector(32 * TILE_H - 1 downto 0);
        ga_done     : out std_logic;

        ---------------------------------------------------------------
        -- qkv_proj_calc
        ---------------------------------------------------------------
        pc_start         : in  std_logic;
        pc_done          : out std_logic;
        pc_bram_addr     : in  integer range 0 to BASE_OFFSET + TOKEN_DIM - 1;
        pc_bram_data_in  : in  std_logic_vector(MHA_DATA_W - 1 downto 0);
        pc_bram_data_out : out std_logic_vector(MHA_DATA_W - 1 downto 0);
        pc_bram_wr       : in  std_logic;
        pc_input_a       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        pc_input_b       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        pc_start_gemm    : out std_logic;
        pc_fetch_gemm    : out std_logic;
        pc_done_gemm     : in  std_logic;

        ---------------------------------------------------------------
        -- qkv_score_calc
        ---------------------------------------------------------------
        sc_start        : in  std_logic;
        sc_done         : out std_logic;
        sc_fetch_qkv    : in  std_logic;
        sc_bram_rd      : in  std_logic;
        sc_bram_addr    : in  integer range 0 to QKV_BRAM_ELEM - 1;
        sc_bram_data_in : in  std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0);

        ---------------------------------------------------------------
        -- qk_score_bram (patched -- see patches/)
        ---------------------------------------------------------------
        qb_fetch_score   : in  std_logic;
        qb_addr_off_1    : in  integer;
        qb_bram_data_in  : in  std_logic_vector(32 * TILE_H - 1 downto 0);
        qb_bram_data_out : out score_io_type;

        ---------------------------------------------------------------
        -- taylor_exp
        ---------------------------------------------------------------
        te_start      : in  std_logic;
        te_done       : out std_logic;
        te_fetch_attn : out std_logic;
        te_data_out   : out std_logic_vector(TILE_H * SCORE_BRAM_DATA_W - 1 downto 0);

        ---------------------------------------------------------------
        -- ffn
        -- NOTE: fn_data_in is 16*17 = 272 bits while taylor_exp emits
        -- TILE_H*SCORE_BRAM_DATA_W = 252. They do not connect. Exposed
        -- separately on purpose -- KNOWN_ISSUES.md section 5.
        ---------------------------------------------------------------
        fn_start         : in  std_logic;
        fn_done          : out std_logic;
        fn_fetch_attn    : in  std_logic;
        fn_data_in       : in  std_logic_vector(16 * 17 - 1 downto 0);
        fn_bram_addr     : in  integer;
        fn_bram_data_in  : in  std_logic_vector(7 downto 0);
        fn_bram_data_out : out std_logic_vector(7 downto 0);
        fn_bram_wr       : in  std_logic;
        fn_start_mac     : out std_logic;
        fn_clear_mac     : out std_logic;
        fn_done_mac      : in  std_logic;
        fn_input_a       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        fn_input_b       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        fn_output_d      : in  std_logic_vector(32 * TILE_H - 1 downto 0);

        ---------------------------------------------------------------
        -- multi_head
        -- NOTE: no result-return port exists on this entity, so the MAC
        -- output has nowhere to land -- KNOWN_ISSUES.md section 4.
        ---------------------------------------------------------------
        mh_start         : in  std_logic;
        mh_done          : out std_logic;
        mh_bram_addr     : in  integer range 0 to (3 * (HEAD_DIM * MODEL_DIM) + TOKEN_DIM) - 1;
        mh_bram_data_in  : in  std_logic_vector(7 downto 0);
        mh_bram_data_out : out std_logic_vector(7 downto 0);
        mh_bram_wr       : in  std_logic;
        mh_input_a       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        mh_input_b       : out int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        mh_start_mac     : out std_logic;
        mh_fetch_mac     : out std_logic;
        mh_done_mac      : in  std_logic;

        ---------------------------------------------------------------
        -- smx_core (score_row driven by qk_score_bram internally)
        ---------------------------------------------------------------
        sx_start           : in  std_logic;
        sx_done            : out std_logic;
        sx_score_row_valid : in  std_logic;
        sx_v_valid         : in  std_logic;
        sx_v_key           : in  integer range 0 to 15;
        sx_v_dim           : in  integer range 0 to HEAD_DIM - 1;
        sx_v_data          : in  signed(7 downto 0);
        sx_attn_valid      : out std_logic;
        sx_attn_dim        : out integer range 0 to HEAD_DIM - 1;
        sx_attn_data       : out signed(31 downto 0);

        ---------------------------------------------------------------
        -- weight_stream_ctrl
        ---------------------------------------------------------------
        ws_start       : in  std_logic;
        ws_done        : out std_logic;
        ws_busy        : out std_logic;
        ws_layer_start : out std_logic;
        ws_layer_done  : in  std_logic;
        ws_layer_index : out integer range 0 to 2;
        ws_buf_sel     : out std_logic;
        ws_mem_addr    : out unsigned(19 downto 0);
        ws_mem_req     : out std_logic;
        ws_mem_data    : in  std_logic_vector(7 downto 0);
        ws_mem_valid   : in  std_logic;
        ws_wr_addr     : out unsigned(19 downto 0);
        ws_wr_data     : out std_logic_vector(7 downto 0);
        ws_wr_en       : out std_logic;
        ws_wr_bank     : out std_logic
    );
end entity;

architecture structural of mirage_sizing_container is

    -- qkv_score_calc <-> gemm_array (real pairing: qkv_score_calc is the
    -- only MAC client whose entity actually carries an output_d input)
    signal sc_input_a    : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal sc_input_b    : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal sc_start_gemm : std_logic;
    signal sc_clear_gemm : std_logic;
    signal gemm_out_d    : std_logic_vector(32 * TILE_H - 1 downto 0);
    signal gemm_done     : std_logic;

    -- qkv_score_calc -> qk_score_bram (name- and type-exact iterators)
    signal qk_iter    : integer;
    signal tok_q_iter : integer;
    signal tok_k_iter : integer;

    -- taylor_exp -> qk_score_bram read side, qk_score_bram -> consumers
    signal te_bram_rd    : std_logic;
    signal te_addr_off_0 : integer;
    signal score_row     : score_io_type;

begin

    ga_output_d      <= gemm_out_d;
    ga_done          <= gemm_done;
    qb_bram_data_out <= score_row;

    ---------------------------------------------------------------
    -- Shared int8 MAC tile. This is the DSP consumer.
    ---------------------------------------------------------------
    gemm_inst : entity work.gemm_array
        port map (
            clk      => clk,
            rst      => rst,
            start    => sc_start_gemm,
            clear    => sc_clear_gemm,
            input_a  => sc_input_a,
            input_b  => sc_input_b,
            output_d => gemm_out_d,
            done     => gemm_done
        );

    ---------------------------------------------------------------
    -- QK^T datapath. Contains qkv_proj_bram + 2x quantizer.
    ---------------------------------------------------------------
    score_calc_inst : entity work.qkv_score_calc
        port map (
            clk          => clk,
            rst          => rst,
            start        => sc_start,
            done         => sc_done,
            fetch_qkv    => sc_fetch_qkv,
            bram_rd      => sc_bram_rd,
            bram_addr    => sc_bram_addr,
            bram_data_in => sc_bram_data_in,
            start_gemm   => sc_start_gemm,
            clear_gemm   => sc_clear_gemm,
            done_gemm    => gemm_done,
            input_a      => sc_input_a,
            input_b      => sc_input_b,
            output_d     => gemm_out_d,
            qk_iter      => qk_iter,
            tok_q_iter   => tok_q_iter,
            tok_k_iter   => tok_k_iter
        );

    ---------------------------------------------------------------
    -- Score BRAM. Feeds both taylor_exp and smx_core (both take
    -- score_io_type -- the pairing is explicit in the source).
    ---------------------------------------------------------------
    score_bram_inst : entity work.qk_score_bram
        port map (
            clk           => clk,
            rst           => rst,
            fetch_score   => qb_fetch_score,
            qk_iter       => qk_iter,
            tok_q_iter    => tok_q_iter,
            tok_k_iter    => tok_k_iter,
            bram_rd       => te_bram_rd,
            addr_off_0    => te_addr_off_0,
            addr_off_1    => qb_addr_off_1,
            bram_data_in  => qb_bram_data_in,
            bram_data_out => score_row
        );

    ---------------------------------------------------------------
    -- Polynomial exp. Drives the score BRAM read port directly --
    -- bram_rd/addr_off_0 out here match bram_rd/addr_off_0 in there.
    ---------------------------------------------------------------
    taylor_inst : entity work.taylor_exp
        port map (
            clk        => clk,
            rst        => rst,
            start      => te_start,
            done       => te_done,
            fetch_attn => te_fetch_attn,
            bram_rd    => te_bram_rd,
            addr_off_0 => te_addr_off_0,
            data_in    => score_row,
            data_out   => te_data_out
        );

    ---------------------------------------------------------------
    -- Softmax top (outputs/ draft). Contains smx_cache + smx_v_calc.
    -- Its own taylor_exp instantiation is commented out upstream, so
    -- taylor_inst above is the exp cost, counted once.
    ---------------------------------------------------------------
    smx_inst : entity work.smx_core
        port map (
            clk             => clk,
            rst             => rst,
            start           => sx_start,
            done            => sx_done,
            score_row_valid => sx_score_row_valid,
            score_row       => score_row,
            v_valid         => sx_v_valid,
            v_key           => sx_v_key,
            v_dim           => sx_v_dim,
            v_data          => sx_v_data,
            attn_valid      => sx_attn_valid,
            attn_dim        => sx_attn_dim,
            attn_data       => sx_attn_data
        );

    ---------------------------------------------------------------
    -- Projection datapath. MAC ports exposed: it has no output_d, and
    -- there is no arbiter for a second gemm client.
    ---------------------------------------------------------------
    proj_calc_inst : entity work.qkv_proj_calc
        port map (
            clk           => clk,
            rst           => rst,
            start         => pc_start,
            done          => pc_done,
            bram_addr     => pc_bram_addr,
            bram_data_in  => pc_bram_data_in,
            bram_data_out => pc_bram_data_out,
            bram_wr       => pc_bram_wr,
            input_a       => pc_input_a,
            input_b       => pc_input_b,
            start_gemm    => pc_start_gemm,
            fetch_gemm    => pc_fetch_gemm,
            done_gemm     => pc_done_gemm
        );

    ---------------------------------------------------------------
    -- MHA control FSM. Same: no result-return port on the entity.
    ---------------------------------------------------------------
    multi_head_inst : entity work.multi_head
        port map (
            clk           => clk,
            rst           => rst,
            start         => mh_start,
            done          => mh_done,
            bram_addr     => mh_bram_addr,
            bram_data_in  => mh_bram_data_in,
            bram_data_out => mh_bram_data_out,
            bram_wr       => mh_bram_wr,
            input_a       => mh_input_a,
            input_b       => mh_input_b,
            start_mac     => mh_start_mac,
            fetch_mac     => mh_fetch_mac,
            done_mac      => mh_done_mac
        );

    ---------------------------------------------------------------
    -- Feed-forward network. MAC ports exposed (second gemm client).
    ---------------------------------------------------------------
    ffn_inst : entity work.ffn
        port map (
            clk           => clk,
            rst           => rst,
            start         => fn_start,
            done          => fn_done,
            fetch_attn    => fn_fetch_attn,
            data_in       => fn_data_in,
            bram_addr     => fn_bram_addr,
            bram_data_in  => fn_bram_data_in,
            bram_data_out => fn_bram_data_out,
            bram_wr       => fn_bram_wr,
            start_mac     => fn_start_mac,
            clear_mac     => fn_clear_mac,
            done_mac      => fn_done_mac,
            input_a       => fn_input_a,
            input_b       => fn_input_b,
            output_d      => fn_output_d
        );

    ---------------------------------------------------------------
    -- DDR weight-stream sequencer.
    ---------------------------------------------------------------
    weight_stream_inst : entity work.weight_stream_ctrl
        port map (
            clk         => clk,
            rst         => rst,
            start       => ws_start,
            done        => ws_done,
            busy        => ws_busy,
            layer_start => ws_layer_start,
            layer_done  => ws_layer_done,
            layer_index => ws_layer_index,
            buf_sel     => ws_buf_sel,
            mem_addr    => ws_mem_addr,
            mem_req     => ws_mem_req,
            mem_data    => ws_mem_data,
            mem_valid   => ws_mem_valid,
            wr_addr     => ws_wr_addr,
            wr_data     => ws_wr_data,
            wr_en       => ws_wr_en,
            wr_bank     => ws_wr_bank
        );

end architecture;
