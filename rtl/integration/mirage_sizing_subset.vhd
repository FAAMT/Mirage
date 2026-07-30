-- =====================================================================
-- mirage_sizing_subset.vhd
--
-- The part of Mirage that can actually be synthesised today.
--
-- Identical in purpose and method to mirage_sizing_container.vhd: each
-- root module once, only genuinely-defined connections wired, everything
-- else exposed as a port. The difference is that this one OMITS the three
-- modules whose weight memories cannot be implemented on any FPGA as
-- written:
--
--   ffn            bram_hlw 1,048,576 b + bram_olw 1,048,576 b + bram_proj 131,072 b
--   multi_head     bram_qkv   786,440 b
--   qkv_proj_calc  bram_qkv   786,440 b
--
-- Each is read inside a 12x5 nested loop -- 60 concurrent read ports.
-- BRAM offers 2, LUTRAM up to 16. Vivado therefore degrades them to
-- registers and errors with [Synth 8-3391] (2026-07-29 run, see runs/).
-- Including them does not produce a large number; it produces no number,
-- and Vivado grinds for a long time before saying so.
--
-- Run ../tools/memory_audit.py for the current classification, and see
-- KNOWN_ISSUES.md section 7 for why this is architecture work rather
-- than something a packaging fix can resolve.
--
-- CAVEAT on the numbers this yields: qkv_score_calc contains
-- qkv_proj_bram, whose bram_qkv (49,152 b, 120 read ports, array-wide
-- reset) also fails BRAM inference. It is under Vivado's dissolve limit,
-- so it will not error -- it will quietly become 49,152 flip-flops on a
-- device with 28,800. Expect FF over 100% and read it as "this memory
-- must become real BRAM", not as a logic cost.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.types_pkg.all;

entity mirage_sizing_subset is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- gemm_array (input side driven by qkv_score_calc internally)
        ga_output_d : out std_logic_vector(32 * TILE_H - 1 downto 0);
        ga_done     : out std_logic;

        -- qkv_score_calc
        sc_start        : in  std_logic;
        sc_done         : out std_logic;
        sc_fetch_qkv    : in  std_logic;
        sc_bram_rd      : in  std_logic;
        sc_bram_addr    : in  integer range 0 to QKV_BRAM_ELEM - 1;
        sc_bram_data_in : in  std_logic_vector(TILE_H * QKV_BRAM_DATA_W - 1 downto 0);

        -- qk_score_bram
        qb_fetch_score   : in  std_logic;
        qb_addr_off_1    : in  integer;
        qb_bram_data_in  : in  std_logic_vector(32 * TILE_H - 1 downto 0);
        qb_bram_data_out : out score_io_type;

        -- taylor_exp
        te_start      : in  std_logic;
        te_done       : out std_logic;
        te_fetch_attn : out std_logic;
        te_data_out   : out std_logic_vector(TILE_H * SCORE_BRAM_DATA_W - 1 downto 0);

        -- smx_core (score_row driven by qk_score_bram internally)
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

        -- weight_stream_ctrl
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

architecture structural of mirage_sizing_subset is

    signal sc_input_a    : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal sc_input_b    : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal sc_start_gemm : std_logic;
    signal sc_clear_gemm : std_logic;
    signal gemm_out_d    : std_logic_vector(32 * TILE_H - 1 downto 0);
    signal gemm_done     : std_logic;

    signal qk_iter    : integer;
    signal tok_q_iter : integer;
    signal tok_k_iter : integer;

    signal te_bram_rd    : std_logic;
    signal te_addr_off_0 : integer;
    signal score_row     : score_io_type;

begin

    ga_output_d      <= gemm_out_d;
    ga_done          <= gemm_done;
    qb_bram_data_out <= score_row;

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
