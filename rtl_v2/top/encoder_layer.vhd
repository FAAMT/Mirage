-- =====================================================================
-- encoder_layer.vhd — the top level v1 never had
--
-- v1's outputs/synth.tcl called `synth_design -top encoder_layer` and no
-- such entity existed anywhere in either repository (KNOWN_ISSUES 3). That
-- is why v1 could only ever be sized per-module, and why the container in
-- rtl/integration had to exist at all.
--
-- This is that entity, and it is FULLY CONNECTED: every submodule port is
-- driven or consumed. That is not tidiness, it is the difference between a
-- resource number and a fiction. v1's smx_core measured SMALLER than its
-- own children (1,668 LUT vs 3,999) because its score_row input was
-- unconnected and Vivado deleted the logic behind it. An unconnected input
-- is not a stub; it is a request to delete the block.
--
-- STRUCTURE
--
--   layer_ctrl ---------- op sequence: Q,K,V -> per head (stage,QK^T,
--        |                softmax,stage,xV) -> O -> FFN1 -> FFN2
--        +-- gemm_sequencer --- the ONE MAC client, walks ct/t/k
--        |        |
--        |   mac_array (32 x DSP48E1)
--        |   weight_window (8 banks x 2 buf = 16 RAMB36)
--        |   act_mem (64 KB = 16 RAMB36, 1r/1w)
--        |
--        +-- stage_engine ----- activation -> weight window, for the two
--        |                      matmuls whose operand is an activation
--        +-- softmax_unit ----- exp_rom + recip_rom, no divider
--
-- RESOURCE INTENT (arithmetic; see V2_DESIGN.md for the derivation)
--   DSP48E1  32 of 66  (48%)   mac_array, one per lane
--   RAMB36   35 of 50  (70%)   16 weight + 16 activation + 3 table/score
--   LUT      control only -- all storage and arithmetic is in hard blocks
--
-- against v1's measured 18,182 LUT (126%), 0 BRAM, 2 DSP.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity encoder_layer is
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        -- host control
        start : in  std_logic;
        n_tok : in  std_logic_vector(4 downto 0);
        busy  : out std_logic;
        done  : out std_logic;

        -- activation load (host or DMA), accepted while idle
        host_we    : in std_logic;
        host_waddr : in std_logic_vector(ACT_AW - 1 downto 0);
        host_wdata : in std_logic_vector(ACT_W - 1 downto 0);

        -- weight stream in from DDR
        wstream_valid : in  std_logic;
        wstream_data  : in  std_logic_vector(BANK_W - 1 downto 0);
        wstream_row   : in  std_logic_vector(BANK_AW - 1 downto 0);
        wstream_bank  : in  std_logic_vector(2 downto 0);
        wstream_buf   : out std_logic;

        -- layer output readback
        out_valid : out std_logic;
        out_data  : out std_logic_vector(ACT_W - 1 downto 0)
    );
end entity;

architecture rtl of encoder_layer is

    -- layer_ctrl <-> sequencer
    signal seq_start, seq_busy, seq_done : std_logic;
    signal k_len     : std_logic_vector(ACT_AW - 1 downto 0);
    signal n_coltile : std_logic_vector(4 downto 0);
    signal act_base  : std_logic_vector(ACT_AW - 1 downto 0);
    signal out_base  : std_logic_vector(ACT_AW - 1 downto 0);
    signal rq_shift  : std_logic_vector(5 downto 0);

    -- routing decode
    signal wb_score, attn_op : std_logic;
    signal sc_waddr_o        : std_logic_vector(SCORE_AW - 1 downto 0);

    -- staging
    signal stage_go, stage_busy, stage_fin : std_logic;
    signal stage_base   : std_logic_vector(ACT_AW - 1 downto 0);
    signal stage_rows   : std_logic_vector(6 downto 0);
    signal stage_lanes  : std_logic_vector(6 downto 0);
    signal stage_xpose  : std_logic;
    signal st_act_re    : std_logic;
    signal st_act_raddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal st_w_we      : std_logic;
    signal st_w_row     : std_logic_vector(BANK_AW - 1 downto 0);
    signal st_w_bank    : std_logic_vector(2 downto 0);
    signal st_w_data    : std_logic_vector(BANK_W - 1 downto 0);

    -- activation memory
    signal sq_act_re    : std_logic;
    signal sq_act_raddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal act_re       : std_logic;
    signal act_raddr    : std_logic_vector(ACT_AW - 1 downto 0);
    signal act_dout     : std_logic_vector(ACT_W - 1 downto 0);
    signal act_data     : act_t;
    signal act_we       : std_logic;
    signal act_waddr    : std_logic_vector(ACT_AW - 1 downto 0);
    signal act_wdata    : std_logic_vector(ACT_W - 1 downto 0);

    -- sequencer writeback
    signal seq_we    : std_logic;
    signal seq_waddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal seq_data  : act_t;
    signal seq_wide  : score_t;

    -- weight window
    signal w_re    : std_logic;
    signal w_row   : std_logic_vector(BANK_AW - 1 downto 0);
    signal w_bus   : wgt_bus_t;
    signal buf_sel : std_logic;
    signal ww_we   : std_logic;
    signal ww_row  : std_logic_vector(BANK_AW - 1 downto 0);
    signal ww_bank : std_logic_vector(2 downto 0);
    signal ww_data : std_logic_vector(BANK_W - 1 downto 0);

    -- MAC array
    signal sq_mac_a : act_t;
    signal mac_a    : act_t;
    signal mac_w    : wgt_bus_t;
    signal mac_en   : std_logic;
    signal mac_clr  : std_logic;
    signal acc_bus  : acc_bus_t;

    -- softmax
    signal sc_we    : std_logic;
    signal sc_wdata : std_logic_vector(SCORE_W - 1 downto 0);
    signal sc_re    : std_logic;
    signal sc_raddr : std_logic_vector(SCORE_AW - 1 downto 0);
    signal sc_dout  : std_logic_vector(SCORE_W - 1 downto 0);

    signal sm_row_start, sm_pass_exp, sm_s_valid : std_logic;
    signal sm_e_valid : std_logic;
    signal sm_e_data  : act_t;
    signal sm_sum8    : std_logic_vector(SUM8_W - 1 downto 0);

    signal rc_re    : std_logic;
    signal rc_dout  : std_logic_vector(RECIP_ROM_DW - 1 downto 0);
    signal mac_scale : std_logic_vector(RECIP_ROM_DW - 1 downto 0);

begin

    ---------------------------------------------------------------------
    -- Activation memory read port: the sequencer owns it, except during a
    -- staging pass when the sequencer is idle by construction.
    ---------------------------------------------------------------------
    act_re    <= st_act_re    when stage_busy = '1' else sq_act_re;
    act_raddr <= st_act_raddr when stage_busy = '1' else sq_act_raddr;

    -- Write port: sequencer writebacks while busy, host loads while idle.
    act_we    <= seq_we    when seq_busy = '1' else host_we;
    act_waddr <= seq_waddr when seq_busy = '1' else host_waddr;
    act_wdata <= std_logic_vector(seq_data) when seq_busy = '1' else host_wdata;

    act_mem : entity work.bram_sdp
        generic map (DATA_W => ACT_W, ADDR_W => ACT_AW, DOUT_REG => false)
        port map (clk => clk,
                  we => act_we, waddr => act_waddr, din => act_wdata,
                  re => act_re, raddr => act_raddr, dout => act_dout);

    act_data  <= signed(act_dout);
    out_data  <= act_dout;
    out_valid <= act_re;

    ---------------------------------------------------------------------
    -- Weight window write port: the DDR stream, or the staging engine.
    -- Staging takes priority because compute is stalled waiting for it.
    ---------------------------------------------------------------------
    ww_we   <= st_w_we   when stage_busy = '1' else wstream_valid;
    ww_row  <= st_w_row  when stage_busy = '1' else wstream_row;
    ww_bank <= st_w_bank when stage_busy = '1' else wstream_bank;
    ww_data <= st_w_data when stage_busy = '1' else wstream_data;

    wgt : entity work.weight_window
        port map (clk => clk,
                  rd_en => w_re, rd_row => w_row, rd_buf => buf_sel, w_bus => w_bus,
                  wr_en => ww_we, wr_row => ww_row,
                  wr_bank => ww_bank, wr_data => ww_data);

    wstream_buf <= buf_sel;

    stage : entity work.stage_engine
        port map (clk => clk, rst => rst,
                  go => stage_go, busy => stage_busy, fin => stage_fin,
                  base => stage_base, rows => stage_rows,
                  lanes => stage_lanes, xpose => stage_xpose,
                  act_re => st_act_re, act_raddr => st_act_raddr,
                  act_data => act_data,
                  w_we => st_w_we, w_row => st_w_row,
                  w_bank => st_w_bank, w_data => st_w_data);

    ---------------------------------------------------------------------
    -- MAC array. For the xV op the broadcast operand is the softmax exp
    -- stream, not activation memory -- that is the whole reason softmax
    -- never materialises probabilities.
    ---------------------------------------------------------------------
    mac_a <= sm_e_data when attn_op = '1' else sq_mac_a;

    mac : entity work.mac_array
        port map (clk => clk, a => mac_a, w_bus => mac_w,
                  en => mac_en, clr => mac_clr, acc_bus => acc_bus);

    ---------------------------------------------------------------------
    -- Softmax normalisation scale: recip_rom[sum8] during xV, unity
    -- otherwise. One multiplier in the sequencer's serialised writeback
    -- divides the whole output vector by the exp-sum.
    ---------------------------------------------------------------------
    mac_scale <= rc_dout when attn_op = '1'
                 else std_logic_vector(to_unsigned(2 ** RECIP_FRAC, RECIP_ROM_DW));

    seq : entity work.gemm_sequencer
        port map (clk => clk, rst => rst,
                  start => seq_start, busy => seq_busy, done => seq_done,
                  k_len => k_len, n_tok => n_tok, n_coltile => n_coltile,
                  act_base => act_base, out_base => out_base,
                  rq_shift => rq_shift, scale => mac_scale,
                  act_re => sq_act_re, act_raddr => sq_act_raddr,
                  act_data => act_data,
                  w_re => w_re, w_row => w_row, w_bus => w_bus,
                  mac_a => sq_mac_a, mac_w => mac_w,
                  mac_en => mac_en, mac_clr => mac_clr, acc_bus => acc_bus,
                  out_we => seq_we, out_waddr => seq_waddr,
                  out_data => seq_data, out_wide => seq_wide);

    ---------------------------------------------------------------------
    -- Score memory. Written by the sequencer's WIDE narrowing during the
    -- QK^T op; read SERIALLY, one address per cycle, by the softmax passes.
    --
    -- That serial read is the entire difference from v1's qk_score_bram,
    -- where 32 concurrent reads of a 2,688-bit array cost 11,587 LUTs --
    -- 80% of the device for 336 bytes of storage.
    ---------------------------------------------------------------------
    sc_we    <= seq_we and wb_score;
    sc_wdata <= std_logic_vector(seq_wide);

    score_mem : entity work.bram_sdp
        generic map (DATA_W => SCORE_W, ADDR_W => SCORE_AW, DOUT_REG => false)
        port map (clk => clk,
                  we => sc_we, waddr => sc_waddr_o, din => sc_wdata,
                  re => sc_re, raddr => sc_raddr, dout => sc_dout);

    smx : entity work.softmax_unit
        port map (clk => clk, rst => rst,
                  row_start => sm_row_start, pass_exp => sm_pass_exp,
                  s_valid => sm_s_valid, s_data => signed(sc_dout),
                  e_valid => sm_e_valid, e_data => sm_e_data, sum8 => sm_sum8);

    recip : entity work.recip_rom
        port map (clk => clk, re => rc_re, addr => sm_sum8, dout => rc_dout);

    ---------------------------------------------------------------------
    -- Op sequencing.
    ---------------------------------------------------------------------
    ctrl : entity work.layer_ctrl
        port map (clk => clk, rst => rst,
                  start => start, n_tok => n_tok, stage_done => stage_fin,
                  busy => busy, done => done,
                  seq_start => seq_start, seq_done => seq_done,
                  k_len => k_len, n_coltile => n_coltile,
                  act_base => act_base, out_base => out_base,
                  rq_shift => rq_shift, buf_sel => buf_sel,
                  sm_row_start => sm_row_start, sm_pass_exp => sm_pass_exp,
                  sm_s_valid => sm_s_valid,
                  sc_re => sc_re, sc_raddr => sc_raddr, rc_re => rc_re,
                  wb_score => wb_score, sc_waddr_o => sc_waddr_o,
                  attn_op => attn_op,
                  stage_go => stage_go, stage_base => stage_base,
                  stage_rows => stage_rows, stage_lanes => stage_lanes,
                  stage_xpose => stage_xpose);

end architecture;
