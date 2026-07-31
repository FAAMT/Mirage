-- =====================================================================
-- mirage_axi_lite.vhd -- AXI4-Lite control port for encoder_layer
--
-- encoder_layer is bit-exact verified. This shim exists so the PS can
-- reach it WITHOUT that verification being reopened: every fabric-facing
-- signal below is driven with the same timing the GHDL command
-- interpreter used, and nothing in encoder_layer changes.
--
-- Three things here are not a "direct map", and each is a defect the
-- naive shim would have:
--
--   1. `done` is a ONE-CYCLE PULSE. layer_ctrl sets done_i in S_DONE and
--      clears it as a default on the next cycle. Software polling a
--      status register over AXI -- best case tens of cycles per read --
--      cannot observe a one-cycle pulse and would hang forever. It is
--      latched here into a sticky bit, cleared by write-1-to-clear, and
--      also driven out as a level interrupt.
--
--   2. Activation and score reads take a cycle. The verified protocol is
--      exactly: drive raddr + re, one rising edge, data valid. But
--      encoder_layer's `out_valid` is a concurrent copy of act_re, so it
--      is high a cycle BEFORE out_data is valid. Using it as a data-valid
--      flag reads the previous byte for the whole transfer. This shim
--      ignores out_valid and counts the cycle itself.
--
--   3. Config registers may not be written while busy -- layer_ctrl reads
--      them combinationally for the whole job. A write during busy is
--      dropped and flagged in ISR[1] rather than silently corrupting a
--      job that is already running.
--
-- Register map (byte offsets; AXI4-Lite, 32-bit, one word per access)
--
--   0x00  ID        R    0x4D495247 "MIRG" -- liveness and version
--   0x04  CTRL      W    [0] start pulse, [2:1] job_op
--   0x08  STATUS    R    [0] busy  [1] done_sticky  [2] wstream_buf
--   0x0C  ISR       RW1C [0] job done  [1] cfg write while busy (error)
--   0x10  ACT_PTR   RW   activation byte pointer
--   0x14  ACT_DATA  RW   W: byte -> [ptr], ptr++ ; R: [ptr] -> byte, ptr++
--   0x18  SC_PTR    RW   score word pointer
--   0x1C  SC_DATA   R    score at [ptr] sign-extended, ptr++
--   0x24  WGT_DATA  W    push one weight word into the stream sink
--   0x28  WGT_LAST  W    push one weight word with TLAST (ends a window)
--   0x40  CFG0 .. 0x68 CFG10   W  -> cfg_addr = (offset - 0x40)/4
--
-- WGT_DATA/WGT_LAST are a deliberate slow path. The weight stream is the
-- bandwidth-critical direction and belongs on a DMA, but requiring one on
-- day one would mean the first board test debugs the accelerator, the
-- DMA, the HP port and the cache policy simultaneously. These two
-- registers push words into the same sink over AXI4-Lite, so bring-up
-- needs nothing but the control port -- slow enough to be useless in
-- production, sufficient to prove the datapath on silicon.
--
-- The auto-incrementing ACT_DATA port is what makes a 512-byte
-- activation vector one loop with no address write per byte. It is still
-- PIO and therefore slow; it is for bring-up and readback, not for the
-- weight stream, which has its own path.
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity mirage_axi_lite is
    port
    (
        -- AXI4-Lite slave, synchronous to the fabric clock
        s_axi_aclk    : in  std_logic;
        s_axi_aresetn : in  std_logic;

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

        irq : out std_logic;   -- level, = ISR[0]

        -- fabric side: encoder_layer pins, driven exactly as the verified
        -- testbench drove them
        rst        : out std_logic;
        job_start  : out std_logic;
        job_op     : out std_logic_vector(1 downto 0);
        busy       : in  std_logic;
        done       : in  std_logic;

        cfg_we     : out std_logic;
        cfg_addr   : out std_logic_vector(3 downto 0);
        cfg_data   : out std_logic_vector(31 downto 0);

        host_we    : out std_logic;
        host_waddr : out std_logic_vector(ACT_AW - 1 downto 0);
        host_wdata : out std_logic_vector(ACT_W - 1 downto 0);
        host_re    : out std_logic;
        host_raddr : out std_logic_vector(ACT_AW - 1 downto 0);
        out_data   : in  std_logic_vector(ACT_W - 1 downto 0);

        host_sc_re    : out std_logic;
        host_sc_raddr : out std_logic_vector(SCORE_AW - 1 downto 0);
        host_sc_dout  : in  std_logic_vector(SCORE_W - 1 downto 0);

        wstream_buf : in std_logic;

        -- PIO weight push, merged with the AXI-Stream port in the top
        wgt_pio_valid : out std_logic;
        wgt_pio_data  : out std_logic_vector(BANK_W - 1 downto 0);
        wgt_pio_last  : out std_logic
    );
end entity;

architecture rtl of mirage_axi_lite is

    -- word index = addr(7 downto 2)
    constant IDX_ID      : natural := 16#00# / 4;
    constant IDX_CTRL    : natural := 16#04# / 4;
    constant IDX_STATUS  : natural := 16#08# / 4;
    constant IDX_ISR     : natural := 16#0C# / 4;
    constant IDX_ACT_PTR : natural := 16#10# / 4;
    constant IDX_ACT_DAT : natural := 16#14# / 4;
    constant IDX_SC_PTR  : natural := 16#18# / 4;
    constant IDX_SC_DAT  : natural := 16#1C# / 4;
    constant IDX_WGT_DAT : natural := 16#24# / 4;
    constant IDX_WGT_LST : natural := 16#28# / 4;
    constant IDX_CFG0    : natural := 16#40# / 4;
    constant IDX_CFG_TOP : natural := 16#68# / 4;

    constant MIRAGE_ID : std_logic_vector(31 downto 0) := x"4D495247";

    -- write channel
    signal awready_i : std_logic := '0';
    signal wready_i  : std_logic := '0';
    signal bvalid_i  : std_logic := '0';
    signal aw_en     : std_logic := '1';
    signal aw_addr   : std_logic_vector(7 downto 0) := (others => '0');

    -- read channel
    type rd_state_t is (RD_IDLE, RD_MEM, RD_CAP, RD_RESP);
    signal rd_state  : rd_state_t := RD_IDLE;
    signal arready_i : std_logic := '0';
    signal rvalid_i  : std_logic := '0';
    signal ar_addr   : std_logic_vector(7 downto 0) := (others => '0');
    signal rdata_i   : std_logic_vector(31 downto 0) := (others => '0');

    -- register state
    signal act_ptr   : unsigned(ACT_AW - 1 downto 0)   := (others => '0');
    signal sc_ptr    : unsigned(SCORE_AW - 1 downto 0) := (others => '0');
    signal isr_done  : std_logic := '0';
    signal isr_cfgerr: std_logic := '0';

    signal wr_fire   : std_logic;
    signal wr_idx    : natural range 0 to 63;
    signal rd_idx    : natural range 0 to 63;

begin

    s_axi_awready <= awready_i;
    s_axi_wready  <= wready_i;
    s_axi_bvalid  <= bvalid_i;
    s_axi_bresp   <= "00";           -- OKAY: no address can error here
    s_axi_arready <= arready_i;
    s_axi_rvalid  <= rvalid_i;
    s_axi_rresp   <= "00";
    s_axi_rdata   <= rdata_i;

    irq <= isr_done;

    -- Reset is active high in this design (layer_ctrl: `if rst = '1'`),
    -- AXI is active low. The whole fabric is held in reset by ARESETN, so
    -- there is one reset domain and no crossing to reason about.
    rst <= not s_axi_aresetn;

    wr_idx   <= to_integer(unsigned(aw_addr(7 downto 2)));
    rd_idx   <= to_integer(unsigned(ar_addr(7 downto 2)));
    wr_fire  <= awready_i and s_axi_awvalid and wready_i and s_axi_wvalid;

    ------------------------------------------------------------------
    -- AXI4-Lite write channel handshake
    ------------------------------------------------------------------
    wr_hs : process (s_axi_aclk)
    begin
        if rising_edge(s_axi_aclk) then
            if s_axi_aresetn = '0' then
                awready_i <= '0';
                wready_i  <= '0';
                bvalid_i  <= '0';
                aw_en     <= '1';
                aw_addr   <= (others => '0');
            else
                -- address and data accepted together, so a write never
                -- half-commits and there is no stale address to track
                if awready_i = '0' and s_axi_awvalid = '1'
                   and s_axi_wvalid = '1' and aw_en = '1' then
                    awready_i <= '1';
                    aw_addr   <= s_axi_awaddr;
                    aw_en     <= '0';
                else
                    awready_i <= '0';
                end if;

                if wready_i = '0' and s_axi_wvalid = '1'
                   and s_axi_awvalid = '1' and aw_en = '1' then
                    wready_i <= '1';
                else
                    wready_i <= '0';
                end if;

                if wr_fire = '1' and bvalid_i = '0' then
                    bvalid_i <= '1';
                elsif bvalid_i = '1' and s_axi_bready = '1' then
                    bvalid_i <= '0';
                    aw_en    <= '1';
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------------------------------
    -- Write decode -> fabric. Every fabric strobe is one cycle wide.
    ------------------------------------------------------------------
    wr_dec : process (s_axi_aclk)
    begin
        if rising_edge(s_axi_aclk) then
            if s_axi_aresetn = '0' then
                job_start  <= '0';
                job_op     <= (others => '0');
                cfg_we     <= '0';
                cfg_addr   <= (others => '0');
                cfg_data   <= (others => '0');
                host_we    <= '0';
                host_waddr <= (others => '0');
                host_wdata <= (others => '0');
                act_ptr    <= (others => '0');
                sc_ptr     <= (others => '0');
                isr_done   <= '0';
                isr_cfgerr <= '0';
                wgt_pio_valid <= '0';
                wgt_pio_data  <= (others => '0');
                wgt_pio_last  <= '0';
            else
                job_start     <= '0';
                cfg_we        <= '0';
                host_we       <= '0';
                wgt_pio_valid <= '0';
                wgt_pio_last  <= '0';

                -- done is a one-cycle pulse; this is the only chance to
                -- see it. Latched unconditionally, ahead of any decode,
                -- so a coincident bus write cannot lose the edge.
                if done = '1' then
                    isr_done <= '1';
                end if;

                if wr_fire = '1' then
                    case wr_idx is

                        when IDX_CTRL =>
                            if s_axi_wdata(0) = '1' then
                                job_start <= '1';
                                job_op    <= s_axi_wdata(2 downto 1);
                            end if;

                        when IDX_ISR =>
                            -- write-1-to-clear
                            if s_axi_wdata(0) = '1' then
                                isr_done <= '0';
                            end if;
                            if s_axi_wdata(1) = '1' then
                                isr_cfgerr <= '0';
                            end if;

                        when IDX_ACT_PTR =>
                            act_ptr <= unsigned(s_axi_wdata(ACT_AW - 1 downto 0));

                        when IDX_ACT_DAT =>
                            host_waddr <= std_logic_vector(act_ptr);
                            host_wdata <= s_axi_wdata(ACT_W - 1 downto 0);
                            host_we    <= '1';
                            act_ptr    <= act_ptr + 1;

                        when IDX_SC_PTR =>
                            sc_ptr <= unsigned(s_axi_wdata(SCORE_AW - 1 downto 0));

                        when IDX_WGT_DAT =>
                            wgt_pio_data  <= s_axi_wdata(BANK_W - 1 downto 0);
                            wgt_pio_valid <= '1';

                        when IDX_WGT_LST =>
                            wgt_pio_data  <= s_axi_wdata(BANK_W - 1 downto 0);
                            wgt_pio_valid <= '1';
                            wgt_pio_last  <= '1';

                        when others =>
                            if wr_idx >= IDX_CFG0 and wr_idx <= IDX_CFG_TOP then
                                -- layer_ctrl reads the config combinationally
                                -- for the whole job, so a write during busy
                                -- would change geometry mid-flight. Drop it
                                -- and say so, rather than corrupt the run.
                                if busy = '1' then
                                    isr_cfgerr <= '1';
                                else
                                    cfg_we   <= '1';
                                    cfg_addr <= std_logic_vector(
                                                  to_unsigned(wr_idx - IDX_CFG0, 4));
                                    cfg_data <= s_axi_wdata;
                                end if;
                            end if;

                    end case;
                end if;

                -- a completed memory read advances its own pointer
                if rd_state = RD_CAP then
                    if rd_idx = IDX_ACT_DAT then
                        act_ptr <= act_ptr + 1;
                    elsif rd_idx = IDX_SC_DAT then
                        sc_ptr <= sc_ptr + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

    ------------------------------------------------------------------
    -- AXI4-Lite read channel.
    --
    -- Register reads answer immediately. Activation and score reads take
    -- the RD_MEM -> RD_CAP detour: re is asserted for exactly one cycle
    -- in RD_MEM, and the fabric's data is valid on the edge into RD_CAP,
    -- which is the same one-edge protocol the verified testbench used.
    ------------------------------------------------------------------
    rd_fsm : process (s_axi_aclk)
    begin
        if rising_edge(s_axi_aclk) then
            if s_axi_aresetn = '0' then
                rd_state      <= RD_IDLE;
                arready_i     <= '0';
                rvalid_i      <= '0';
                ar_addr       <= (others => '0');
                rdata_i       <= (others => '0');
                host_re       <= '0';
                host_raddr    <= (others => '0');
                host_sc_re    <= '0';
                host_sc_raddr <= (others => '0');
            else
                arready_i  <= '0';
                host_re    <= '0';
                host_sc_re <= '0';

                case rd_state is

                    when RD_IDLE =>
                        if s_axi_arvalid = '1' and arready_i = '0' then
                            arready_i <= '1';
                            ar_addr   <= s_axi_araddr;

                            if unsigned(s_axi_araddr(7 downto 2)) = IDX_ACT_DAT then
                                host_raddr <= std_logic_vector(act_ptr);
                                host_re    <= '1';
                                rd_state   <= RD_MEM;
                            elsif unsigned(s_axi_araddr(7 downto 2)) = IDX_SC_DAT then
                                host_sc_raddr <= std_logic_vector(sc_ptr);
                                host_sc_re    <= '1';
                                rd_state      <= RD_MEM;
                            else
                                rd_state <= RD_RESP;
                            end if;
                        end if;

                    when RD_MEM =>
                        -- re was high through the edge that entered this
                        -- state; the fabric registers its output on the
                        -- edge that leaves it
                        rd_state <= RD_CAP;

                    when RD_CAP =>
                        if rd_idx = IDX_ACT_DAT then
                            rdata_i <= std_logic_vector(
                                         resize(signed(out_data), 32));
                        else
                            rdata_i <= std_logic_vector(
                                         resize(signed(host_sc_dout), 32));
                        end if;
                        rd_state <= RD_RESP;

                    when RD_RESP =>
                        if rvalid_i = '0' then
                            case rd_idx is
                                when IDX_ID =>
                                    rdata_i <= MIRAGE_ID;
                                when IDX_STATUS =>
                                    rdata_i <= (0 => busy,
                                                1 => isr_done,
                                                2 => wstream_buf,
                                                others => '0');
                                when IDX_ISR =>
                                    rdata_i <= (0 => isr_done,
                                                1 => isr_cfgerr,
                                                others => '0');
                                when IDX_ACT_PTR =>
                                    rdata_i <= std_logic_vector(
                                                 resize(act_ptr, 32));
                                when IDX_SC_PTR =>
                                    rdata_i <= std_logic_vector(
                                                 resize(sc_ptr, 32));
                                when IDX_ACT_DAT | IDX_SC_DAT =>
                                    null;   -- already captured in RD_CAP
                                when others =>
                                    rdata_i <= (others => '0');
                            end case;
                            rvalid_i <= '1';
                        elsif s_axi_rready = '1' then
                            rvalid_i <= '0';
                            rd_state <= RD_IDLE;
                        end if;

                end case;
            end if;
        end if;
    end process;

end architecture;
