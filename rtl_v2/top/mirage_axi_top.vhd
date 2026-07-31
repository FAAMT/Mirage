-- =====================================================================
-- mirage_axi_top.vhd -- the block the Zynq PS actually sees
--
-- Structure, and the reason for it:
--
--     s_axi_ctrl  (AXI4-Lite) --> mirage_axi_lite --+
--                                                   |--> encoder_layer
--     s_axis_wgt  (AXI-Stream) --> wgt_stream ------+     (UNMODIFIED)
--
-- encoder_layer is instantiated exactly as verified. Its 61,632 checked
-- values and the "Hello" -> مرحبا result are properties of that entity,
-- and they stay properties of it only if wrapping does not touch it. All
-- adaptation lives in the two shims, which is also why they are the only
-- things that need new tests.
--
-- Two ports, not one, because the two directions have nothing in common:
-- control is a handful of 32-bit words at human rates, while the weight
-- path needs a sustained 400 MB/s and is the system bottleneck at roughly
-- 4.5 : 1 against compute. Putting weights through AXI4-Lite would work
-- and would be about two orders of magnitude too slow.
--
-- OUT-OF-CONTEXT NOTE. Synthesised standalone this has ~180 port bits and
-- will fail IO placement on clg400 exactly as encoder_layer did -- that
-- is expected and is not a defect. It is meant to sit inside a block
-- design where all of these connect to the PS. tcl/synth_v2.tcl passes
-- -mode out_of_context for the standalone characterisation run.
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity mirage_axi_top is
    port
    (
        -- one clock and one reset for the whole block; the PS drives both
        aclk    : in std_logic;
        aresetn : in std_logic;

        -- AXI4-Lite control
        s_axi_awaddr  : in  std_logic_vector(7 downto 0);
        s_axi_awvalid : in  std_logic;
        s_axi_awready : out std_logic;
        s_axi_wdata   : in  std_logic_vector(31 downto 0);
        s_axi_wstrb   : in  std_logic_vector(3 downto 0);
        s_axi_wvalid  : in  std_logic;
        s_axi_wready  : out std_logic;
        s_axi_bresp   : out std_logic_vector(1 downto 0);
        s_axi_bvalid  : out std_logic;
        s_axi_bready  : in  std_logic;
        s_axi_araddr  : in  std_logic_vector(7 downto 0);
        s_axi_arvalid : in  std_logic;
        s_axi_arready : out std_logic;
        s_axi_rdata   : out std_logic_vector(31 downto 0);
        s_axi_rresp   : out std_logic_vector(1 downto 0);
        s_axi_rvalid  : out std_logic;
        s_axi_rready  : in  std_logic;

        -- AXI4-Stream weight input
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;
        s_axis_tdata  : in  std_logic_vector(BANK_W - 1 downto 0);
        s_axis_tlast  : in  std_logic;

        irq : out std_logic
    );
end entity;

architecture rtl of mirage_axi_top is

    signal rst : std_logic;

    signal job_start : std_logic;
    signal job_op    : std_logic_vector(1 downto 0);
    signal busy      : std_logic;
    signal done      : std_logic;

    signal cfg_we   : std_logic;
    signal cfg_addr : std_logic_vector(3 downto 0);
    signal cfg_data : std_logic_vector(31 downto 0);

    signal host_we    : std_logic;
    signal host_waddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal host_wdata : std_logic_vector(ACT_W - 1 downto 0);
    signal host_re    : std_logic;
    signal host_raddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal out_data   : std_logic_vector(ACT_W - 1 downto 0);
    signal out_valid  : std_logic;

    signal host_sc_re    : std_logic;
    signal host_sc_raddr : std_logic_vector(SCORE_AW - 1 downto 0);
    signal host_sc_dout  : std_logic_vector(SCORE_W - 1 downto 0);

    signal wstream_valid : std_logic;
    signal wstream_data  : std_logic_vector(BANK_W - 1 downto 0);
    signal wstream_row   : std_logic_vector(BANK_AW - 1 downto 0);
    signal wstream_bank  : std_logic_vector(2 downto 0);
    signal wstream_buf   : std_logic;

    signal beat_count : std_logic_vector(31 downto 0);

    -- PIO weight push from the control port, merged with the stream
    signal pio_valid : std_logic;
    signal pio_data  : std_logic_vector(BANK_W - 1 downto 0);
    signal pio_last  : std_logic;
    signal wgt_tvalid, wgt_tlast : std_logic;
    signal wgt_tdata : std_logic_vector(BANK_W - 1 downto 0);

begin

    -- Two producers, one sink. PIO wins when both present, which cannot
    -- happen in practice: bring-up uses PIO with no DMA configured, and
    -- production uses the DMA and never writes WGT_DATA. The priority is
    -- stated rather than left to chance so the failure mode is a dropped
    -- DMA beat, not two words written to one address.
    wgt_tvalid <= pio_valid or s_axis_tvalid;
    wgt_tdata  <= pio_data when pio_valid = '1' else s_axis_tdata;
    wgt_tlast  <= pio_last when pio_valid = '1' else s_axis_tlast;

    ctrl : entity work.mirage_axi_lite
        port map (
            s_axi_aclk    => aclk,
            s_axi_aresetn => aresetn,
            s_axi_awaddr  => s_axi_awaddr,
            s_axi_awvalid => s_axi_awvalid,
            s_axi_awready => s_axi_awready,
            s_axi_wdata   => s_axi_wdata,
            s_axi_wstrb   => s_axi_wstrb,
            s_axi_wvalid  => s_axi_wvalid,
            s_axi_wready  => s_axi_wready,
            s_axi_bresp   => s_axi_bresp,
            s_axi_bvalid  => s_axi_bvalid,
            s_axi_bready  => s_axi_bready,
            s_axi_araddr  => s_axi_araddr,
            s_axi_arvalid => s_axi_arvalid,
            s_axi_arready => s_axi_arready,
            s_axi_rdata   => s_axi_rdata,
            s_axi_rresp   => s_axi_rresp,
            s_axi_rvalid  => s_axi_rvalid,
            s_axi_rready  => s_axi_rready,
            irq           => irq,

            rst        => rst,
            job_start  => job_start,
            job_op     => job_op,
            busy       => busy,
            done       => done,
            cfg_we     => cfg_we,
            cfg_addr   => cfg_addr,
            cfg_data   => cfg_data,
            host_we    => host_we,
            host_waddr => host_waddr,
            host_wdata => host_wdata,
            host_re    => host_re,
            host_raddr => host_raddr,
            out_data   => out_data,

            host_sc_re    => host_sc_re,
            host_sc_raddr => host_sc_raddr,
            host_sc_dout  => host_sc_dout,
            wstream_buf   => wstream_buf,

            wgt_pio_valid => pio_valid,
            wgt_pio_data  => pio_data,
            wgt_pio_last  => pio_last
        );

    wgt : entity work.wgt_stream
        port map (
            clk           => aclk,
            rst           => rst,
            s_axis_tvalid => wgt_tvalid,
            s_axis_tready => s_axis_tready,
            s_axis_tdata  => wgt_tdata,
            s_axis_tlast  => wgt_tlast,
            wstream_valid => wstream_valid,
            wstream_data  => wstream_data,
            wstream_row   => wstream_row,
            wstream_bank  => wstream_bank,
            beat_count    => beat_count
        );

    core : entity work.encoder_layer
        port map (
            clk => aclk,
            rst => rst,

            job_start => job_start,
            job_op    => job_op,
            busy      => busy,
            done      => done,

            cfg_we   => cfg_we,
            cfg_addr => cfg_addr,
            cfg_data => cfg_data,

            host_we    => host_we,
            host_waddr => host_waddr,
            host_wdata => host_wdata,
            host_re    => host_re,
            host_raddr => host_raddr,

            wstream_valid => wstream_valid,
            wstream_data  => wstream_data,
            wstream_row   => wstream_row,
            wstream_bank  => wstream_bank,
            wstream_buf   => wstream_buf,

            out_valid => out_valid,
            out_data  => out_data,

            host_sc_re    => host_sc_re,
            host_sc_raddr => host_sc_raddr,
            host_sc_dout  => host_sc_dout
        );

end architecture;
