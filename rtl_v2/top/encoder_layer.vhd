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
--   layer_ctrl ---------- job dispatcher, driven by the PS one job at a
--        |                time (GEMM / STAGE / SMAX)
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
-- WHO DRIVES WHAT
--
-- The PS owns the op order, the weight reloads and every O(D) elementwise
-- pass (bias, residual, layer norm, the FFN activation function). The PL
-- owns every O(D^2) matmul and the softmax. That split is not a compromise
-- forced by area -- it is where the arithmetic actually is. One encoder
-- layer is 3.15 M multiply-accumulates per token (4 x 512^2 for the
-- projections, 2 x 512 x 2048 for the FFN); the elementwise work is
-- ~13 K operations, 0.4%, and it is the part where int8 dynamic range
-- costs the most accuracy.
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

        -- job control (from the PS)
        job_start : in  std_logic;
        job_op    : in  std_logic_vector(1 downto 0);
        busy      : out std_logic;
        done      : out std_logic;

        -- job configuration register file
        cfg_we   : in std_logic;
        cfg_addr : in std_logic_vector(3 downto 0);
        cfg_data : in std_logic_vector(31 downto 0);

        -- activation load / readback (host or DMA), accepted while idle
        host_we    : in std_logic;
        host_waddr : in std_logic_vector(ACT_AW - 1 downto 0);
        host_wdata : in std_logic_vector(ACT_W - 1 downto 0);
        host_re    : in std_logic;
        host_raddr : in std_logic_vector(ACT_AW - 1 downto 0);

        -- weight stream in from DDR
        wstream_valid : in  std_logic;
        wstream_data  : in  std_logic_vector(BANK_W - 1 downto 0);
        wstream_row   : in  std_logic_vector(BANK_AW - 1 downto 0);
        wstream_bank  : in  std_logic_vector(2 downto 0);
        wstream_buf   : out std_logic;

        -- layer output readback
        out_valid : out std_logic;
        out_data  : out std_logic_vector(ACT_W - 1 downto 0);

        -- Score-memory readback. Verification only: it lets a testbench
        -- check QK^T on its own instead of inferring it from the attention
        -- output four ops later. Costs one 2:1 mux on an address that is
        -- otherwise idle whenever the softmax is not running.
        host_sc_re    : in  std_logic;
        host_sc_raddr : in  std_logic_vector(SCORE_AW - 1 downto 0);
        host_sc_dout  : out std_logic_vector(SCORE_W - 1 downto 0)
    );
end entity;

architecture rtl of encoder_layer is

    -- layer_ctrl <-> sequencer
    signal seq_start, seq_busy, seq_done : std_logic;
    signal k_len     : std_logic_vector(ACT_AW - 1 downto 0);
    signal seq_ntok  : std_logic_vector(4 downto 0);
    signal n_coltile : std_logic_vector(4 downto 0);
    signal act_base  : std_logic_vector(ACT_AW - 1 downto 0);
    signal out_base  : std_logic_vector(ACT_AW - 1 downto 0);
    signal rq_shift  : std_logic_vector(5 downto 0);
    signal acc_clr   : std_logic;
    signal acc_wb    : std_logic;

    -- routing decode
    signal wb_score, use_recip : std_logic;
    signal sc_wbase            : std_logic_vector(SCORE_AW - 1 downto 0);

    -- staging
    signal stage_go, stage_busy, stage_fin : std_logic;
    signal stage_base   : std_logic_vector(ACT_AW - 1 downto 0);
    signal stage_rows   : std_logic_vector(6 downto 0);
    signal stage_lanes  : std_logic_vector(6 downto 0);
    signal stage_stride : std_logic_vector(ACT_AW - 1 downto 0);
    signal stage_xpose  : std_logic;
    signal stage_rowofs : std_logic_vector(BANK_AW - 1 downto 0);
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
    signal mac_w    : wgt_bus_t;
    signal mac_en   : std_logic;
    signal mac_clr  : std_logic;
    signal acc_bus  : acc_bus_t;

    -- softmax
    signal sc_we    : std_logic;
    signal sc_wdata : std_logic_vector(SCORE_W - 1 downto 0);
    signal sc_waddr : std_logic_vector(SCORE_AW - 1 downto 0);
    signal sc_re    : std_logic;
    signal sc_raddr : std_logic_vector(SCORE_AW - 1 downto 0);
    signal sc_dout  : std_logic_vector(SCORE_W - 1 downto 0);
    signal sc_re_mux    : std_logic;
    signal sc_raddr_mux : std_logic_vector(SCORE_AW - 1 downto 0);
    signal busy_i       : std_logic;

    signal sm_row_start, sm_pass_exp, sm_s_valid : std_logic;
    signal sm_e_valid : std_logic;
    signal sm_e_data  : act_t;
    signal sm_e_data_q : act_t := (others => '0');
    signal sm_sum8    : std_logic_vector(SUM8_W - 1 downto 0);

    signal exp_we    : std_logic;
    signal exp_waddr : std_logic_vector(ACT_AW - 1 downto 0);

    signal rc_re     : std_logic;
    signal rc_dout   : std_logic_vector(RECIP_ROM_DW - 1 downto 0);
    signal mac_scale : std_logic_vector(RECIP_ROM_DW - 1 downto 0);
    signal rq_scale  : std_logic_vector(RECIP_ROM_DW - 1 downto 0);

begin

    busy <= busy_i;

    ---------------------------------------------------------------------
    -- Activation memory read port. The sequencer owns it, except during a
    -- staging pass when the sequencer is idle by construction, and except
    -- when the PS is reading a result back out while nothing is running.
    ---------------------------------------------------------------------
    act_re    <= st_act_re    when stage_busy = '1' else
                 sq_act_re    when seq_busy   = '1' else host_re;
    act_raddr <= st_act_raddr when stage_busy = '1' else
                 sq_act_raddr when seq_busy   = '1' else host_raddr;

    ---------------------------------------------------------------------
    -- Activation memory write port. Three writers, none concurrent:
    --   the sequencer's requantised matmul result while it is running,
    --   the softmax's exp row during a SMAX job,
    --   the PS loading activations while the layer is idle.
    ---------------------------------------------------------------------
    act_we    <= seq_we when seq_busy = '1' else
                 exp_we when exp_we   = '1' else host_we;
    act_waddr <= seq_waddr when seq_busy = '1' else
                 exp_waddr when exp_we   = '1' else host_waddr;
    act_wdata <= std_logic_vector(seq_data)    when seq_busy = '1' else
                 std_logic_vector(sm_e_data_q) when exp_we   = '1' else host_wdata;

    -- layer_ctrl registers exp_we and exp_waddr from sm_e_valid, so the
    -- data has to be registered with them. Taking sm_e_data live wrote
    -- e[i+1] at address i and the whole exp row came out shifted one place
    -- left -- caught by model_ref/test_ops.py's exp_row checks.
    exp_align : process (clk)
    begin
        if rising_edge(clk) then
            sm_e_data_q <= sm_e_data;
        end if;
    end process;

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
                  lanes => stage_lanes, stride => stage_stride,
                  xpose => stage_xpose, row_ofs => stage_rowofs,
                  act_re => st_act_re, act_raddr => st_act_raddr,
                  act_data => act_data,
                  w_we => st_w_we, w_row => st_w_row,
                  w_bank => st_w_bank, w_data => st_w_data);

    ---------------------------------------------------------------------
    -- MAC array. The broadcast operand is always activation memory now:
    -- the exp stream reaches it through REG_EXP rather than through a mux,
    -- because producing it in one op and consuming it in another never
    -- worked -- see layer_ctrl's header.
    ---------------------------------------------------------------------
    mac : entity work.mac_array
        port map (clk => clk, a => sq_mac_a, w_bus => mac_w,
                  en => mac_en, clr => mac_clr, acc_bus => acc_bus);

    ---------------------------------------------------------------------
    -- Softmax normalisation scale: recip_rom[sum8] for the xV matmul,
    -- unity otherwise. One multiplier in the sequencer's serialised
    -- writeback divides the whole output vector by the exp-sum.
    ---------------------------------------------------------------------
    mac_scale <= rc_dout when use_recip = '1' else rq_scale;

    seq : entity work.gemm_sequencer
        port map (clk => clk, rst => rst,
                  start => seq_start, busy => seq_busy, done => seq_done,
                  k_len => k_len, n_tok => seq_ntok, n_coltile => n_coltile,
                  act_base => act_base, out_base => out_base,
                  rq_shift => rq_shift, scale => mac_scale,
                  acc_clr => acc_clr, acc_wb => acc_wb,
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
    --
    -- The write address is the head's base plus the sequencer's own
    -- writeback offset, so there is one address generator rather than two
    -- that have to agree.
    ---------------------------------------------------------------------
    sc_we    <= seq_we and wb_score;
    sc_wdata <= std_logic_vector(seq_wide);
    sc_waddr <= std_logic_vector(unsigned(sc_wbase)
                                 + unsigned(seq_waddr(SCORE_AW - 1 downto 0)));

    sc_re_mux    <= sc_re    when busy_i = '1' else host_sc_re;
    sc_raddr_mux <= sc_raddr when busy_i = '1' else host_sc_raddr;

    score_mem : entity work.bram_sdp
        generic map (DATA_W => SCORE_W, ADDR_W => SCORE_AW, DOUT_REG => false)
        port map (clk => clk,
                  we => sc_we, waddr => sc_waddr, din => sc_wdata,
                  re => sc_re_mux, raddr => sc_raddr_mux, dout => sc_dout);

    host_sc_dout <= sc_dout;

    smx : entity work.softmax_unit
        port map (clk => clk, rst => rst,
                  row_start => sm_row_start, pass_exp => sm_pass_exp,
                  s_valid => sm_s_valid, s_data => signed(sc_dout),
                  e_valid => sm_e_valid, e_data => sm_e_data, sum8 => sm_sum8);

    recip : entity work.recip_rom
        port map (clk => clk, re => rc_re, addr => sm_sum8, dout => rc_dout);

    ---------------------------------------------------------------------
    -- Job dispatch.
    ---------------------------------------------------------------------
    ctrl : entity work.layer_ctrl
        port map (clk => clk, rst => rst,
                  job_start => job_start, job_op => job_op,
                  busy => busy_i, done => done,
                  cfg_we => cfg_we, cfg_addr => cfg_addr, cfg_data => cfg_data,
                  seq_start => seq_start, seq_done => seq_done,
                  k_len => k_len, n_tok_o => seq_ntok, n_coltile => n_coltile,
                  act_base => act_base, out_base => out_base,
                  rq_shift => rq_shift, acc_clr => acc_clr, acc_wb => acc_wb,
                  stage_go => stage_go, stage_done => stage_fin,
                  stage_base => stage_base, stage_rows => stage_rows,
                  stage_lanes => stage_lanes, stage_stride => stage_stride,
                  stage_xpose => stage_xpose, stage_rowofs => stage_rowofs,
                  sm_row_start => sm_row_start, sm_pass_exp => sm_pass_exp,
                  sm_s_valid => sm_s_valid, sm_e_valid => sm_e_valid,
                  sc_re => sc_re, sc_raddr => sc_raddr, rc_re => rc_re,
                  exp_we => exp_we, exp_waddr => exp_waddr,
                  wb_score => wb_score, sc_waddr_o => sc_wbase,
                  use_recip => use_recip, rq_scale => rq_scale,
                  buf_sel => buf_sel);

end architecture;
