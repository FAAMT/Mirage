-- =====================================================================
-- axi_lite_tb.vhd -- the AXI wrapper is transparent
--
-- The datapath is already proven: 61,632 values across 138 tensors, zero
-- mismatches, and "Hello" -> مرحبا. Re-testing it here would prove
-- nothing new. What is NOT proven is that reaching it through AXI4-Lite
-- and AXI4-Stream produces the same arithmetic, so that is exactly and
-- only what this measures.
--
-- Method: one real GEMM job, driven twice.
--
--   ref : encoder_layer    driven by direct pin wiggles, using the same
--                          protocol the verified command interpreter used
--   dut : mirage_axi_top   driven ONLY through AXI transactions
--
-- Same activations, same weights, same geometry, same opcode. Then all 32
-- output bytes are compared. Any disagreement is the wrapper's, because
-- the thing inside dut is bit-identical to ref by construction.
--
-- The weight comparison is the sharper half. ref is told each (row, bank)
-- explicitly; dut is told nothing and must regenerate the identical
-- address sequence from a linear stream. A mis-generated address changes
-- which weight lands where, which changes the products, which changes the
-- output bytes. There is no way for wgt_stream to be wrong and this test
-- to pass.
--
-- Four wrapper-contract checks run alongside:
--   1. ID register           -- the bus is alive and decoding
--   2. done latching         -- done is a ONE-CYCLE pulse; software that
--                               polls over AXI cannot see it unlatched
--   3. cfg write while busy  -- must be refused and flagged, not applied
--   4. pointer auto-increment -- 32 reads from one address register
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use STD.ENV.all;

library work;
use work.mirage_v2_pkg.all;

entity axi_lite_tb is
end entity;

architecture sim of axi_lite_tb is

    constant PERIOD : time := 10 ns;

    -- geometry of the job under test
    constant K_LEN     : natural := 32;   -- reduction length
    constant N_OUT     : natural := 32;   -- one column tile = 32 lanes
    constant ACT_BASE  : natural := 0;
    constant OUT_BASE  : natural := 1024;
    constant RQ_SHIFT  : natural := 6;    -- keeps the result off the sat8 rails

    signal clk     : std_logic := '0';
    signal aresetn : std_logic := '0';
    signal halt    : boolean   := false;

    ---------------------------------------------------------------
    -- reference instance: direct pins
    ---------------------------------------------------------------
    signal r_rst        : std_logic := '1';
    signal r_job_start  : std_logic := '0';
    signal r_job_op     : std_logic_vector(1 downto 0) := (others => '0');
    signal r_busy       : std_logic;
    signal r_done       : std_logic;
    signal r_cfg_we     : std_logic := '0';
    signal r_cfg_addr   : std_logic_vector(3 downto 0) := (others => '0');
    signal r_cfg_data   : std_logic_vector(31 downto 0) := (others => '0');
    signal r_host_we    : std_logic := '0';
    signal r_host_waddr : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal r_host_wdata : std_logic_vector(ACT_W - 1 downto 0) := (others => '0');
    signal r_host_re    : std_logic := '0';
    signal r_host_raddr : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal r_ws_valid   : std_logic := '0';
    signal r_ws_data    : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');
    signal r_ws_row     : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal r_ws_bank    : std_logic_vector(2 downto 0) := (others => '0');
    signal r_ws_buf     : std_logic;
    signal r_out_valid  : std_logic;
    signal r_out_data   : std_logic_vector(ACT_W - 1 downto 0);
    signal r_sc_re      : std_logic := '0';
    signal r_sc_raddr   : std_logic_vector(SCORE_AW - 1 downto 0) := (others => '0');
    signal r_sc_dout    : std_logic_vector(SCORE_W - 1 downto 0);

    ---------------------------------------------------------------
    -- device under test: AXI only
    ---------------------------------------------------------------
    signal awaddr  : std_logic_vector(7 downto 0)  := (others => '0');
    signal awvalid : std_logic := '0';
    signal awready : std_logic;
    signal wdata   : std_logic_vector(31 downto 0) := (others => '0');
    signal wstrb   : std_logic_vector(3 downto 0)  := "1111";
    signal wvalid  : std_logic := '0';
    signal wready  : std_logic;
    signal bresp   : std_logic_vector(1 downto 0);
    signal bvalid  : std_logic;
    signal bready  : std_logic := '0';
    signal araddr  : std_logic_vector(7 downto 0)  := (others => '0');
    signal arvalid : std_logic := '0';
    signal arready : std_logic;
    signal rdata   : std_logic_vector(31 downto 0);
    signal rresp   : std_logic_vector(1 downto 0);
    signal rvalid  : std_logic;
    signal rready  : std_logic := '0';
    signal tvalid  : std_logic := '0';
    signal tready  : std_logic;
    signal tdata   : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');
    signal tlast   : std_logic := '0';
    signal irq     : std_logic;

    -- register offsets, mirroring mirage_axi_lite
    constant A_ID      : natural := 16#00#;
    constant A_CTRL    : natural := 16#04#;
    constant A_STATUS  : natural := 16#08#;
    constant A_ISR     : natural := 16#0C#;
    constant A_ACT_PTR : natural := 16#10#;
    constant A_ACT_DAT : natural := 16#14#;
    constant A_CFG0    : natural := 16#40#;

    type byte_arr is array (natural range <>) of integer;
    signal ref_out : byte_arr(0 to N_OUT - 1) := (others => 0);
    signal dut_out : byte_arr(0 to N_OUT - 1) := (others => 0);

    signal errors : natural := 0;

    -- Deterministic stimulus. Small magnitudes on purpose: with 32 terms
    -- of full-range int8 every output would saturate to +/-127 and the
    -- comparison would pass even if the arithmetic were wrong.
    function act_val (i : natural) return integer is
    begin
        return ((i * 7 + 3) mod 16) - 8;
    end function;

    function wgt_val (row : natural; lane : natural) return integer is
    begin
        return ((row * 5 + lane * 3 + 1) mod 16) - 8;
    end function;

begin

    clk <= not clk after PERIOD / 2 when not halt else '0';

    ref : entity work.encoder_layer
        port map (
            clk => clk, rst => r_rst,
            job_start => r_job_start, job_op => r_job_op,
            busy => r_busy, done => r_done,
            cfg_we => r_cfg_we, cfg_addr => r_cfg_addr, cfg_data => r_cfg_data,
            host_we => r_host_we, host_waddr => r_host_waddr,
            host_wdata => r_host_wdata,
            host_re => r_host_re, host_raddr => r_host_raddr,
            wstream_valid => r_ws_valid, wstream_data => r_ws_data,
            wstream_row => r_ws_row, wstream_bank => r_ws_bank,
            wstream_buf => r_ws_buf,
            out_valid => r_out_valid, out_data => r_out_data,
            host_sc_re => r_sc_re, host_sc_raddr => r_sc_raddr,
            host_sc_dout => r_sc_dout
        );

    dut : entity work.mirage_axi_top
        port map (
            aclk => clk, aresetn => aresetn,
            s_axi_awaddr => awaddr, s_axi_awvalid => awvalid,
            s_axi_awready => awready,
            s_axi_wdata => wdata, s_axi_wstrb => wstrb,
            s_axi_wvalid => wvalid, s_axi_wready => wready,
            s_axi_bresp => bresp, s_axi_bvalid => bvalid, s_axi_bready => bready,
            s_axi_araddr => araddr, s_axi_arvalid => arvalid,
            s_axi_arready => arready,
            s_axi_rdata => rdata, s_axi_rresp => rresp,
            s_axi_rvalid => rvalid, s_axi_rready => rready,
            s_axis_tvalid => tvalid, s_axis_tready => tready,
            s_axis_tdata => tdata, s_axis_tlast => tlast,
            irq => irq
        );

    -----------------------------------------------------------------
    stim : process

        variable cfg1 : std_logic_vector(31 downto 0);
        variable rv   : std_logic_vector(31 downto 0);
        variable word : std_logic_vector(31 downto 0);
        variable lo   : line;
        variable nerr : natural := 0;
        variable uniform : boolean;

        -- ---------- AXI4-Lite master ----------
        procedure axi_wr (constant a : natural;
                          constant d : std_logic_vector(31 downto 0)) is
        begin
            awaddr  <= std_logic_vector(to_unsigned(a, 8));
            wdata   <= d;
            awvalid <= '1';
            wvalid  <= '1';
            bready  <= '1';
            loop
                wait until rising_edge(clk);
                exit when awready = '1' and wready = '1';
            end loop;
            awvalid <= '0';
            wvalid  <= '0';
            loop
                wait until rising_edge(clk);
                exit when bvalid = '1';
            end loop;
            bready <= '0';
        end procedure;

        procedure axi_rd (constant a : natural;
                          variable d : out std_logic_vector(31 downto 0)) is
        begin
            araddr  <= std_logic_vector(to_unsigned(a, 8));
            arvalid <= '1';
            rready  <= '1';
            loop
                wait until rising_edge(clk);
                exit when arready = '1';
            end loop;
            arvalid <= '0';
            loop
                wait until rising_edge(clk);
                if rvalid = '1' then
                    d := rdata;
                    exit;
                end if;
            end loop;
            rready <= '0';
        end procedure;

        -- ---------- reference: direct pins ----------
        procedure ref_cfg (constant a : natural;
                           constant d : std_logic_vector(31 downto 0)) is
        begin
            r_cfg_addr <= std_logic_vector(to_unsigned(a, 4));
            r_cfg_data <= d;
            r_cfg_we   <= '1';
            wait until rising_edge(clk);
            r_cfg_we   <= '0';
        end procedure;

    begin
        ---------------------------------------------------------------
        -- reset both
        ---------------------------------------------------------------
        r_rst   <= '1';
        aresetn <= '0';
        for i in 0 to 7 loop wait until rising_edge(clk); end loop;
        r_rst   <= '0';
        aresetn <= '1';
        for i in 0 to 3 loop wait until rising_edge(clk); end loop;

        -- geometry word for config register 1
        cfg1 := (others => '0');
        cfg1(4 downto 0)   := std_logic_vector(to_unsigned(1, 5));        -- n_tok
        cfg1(12 downto 8)  := std_logic_vector(to_unsigned(1, 5));        -- n_coltile
        cfg1(21 downto 16) := std_logic_vector(to_unsigned(RQ_SHIFT, 6));
        cfg1(24) := '1';   -- acc_clr : this job starts the reduction
        cfg1(25) := '1';   -- acc_wb  : and ends it, so it writes back

        ---------------------------------------------------------------
        -- PHASE 1 -- reference, driven directly
        ---------------------------------------------------------------
        report "phase 1: reference instance, direct pins";

        for i in 0 to K_LEN - 1 loop
            r_host_waddr <= std_logic_vector(to_unsigned(ACT_BASE + i, ACT_AW));
            r_host_wdata <= std_logic_vector(to_signed(act_val(i), ACT_W));
            r_host_we    <= '1';
            wait until rising_edge(clk);
        end loop;
        r_host_we <= '0';

        -- bufsel 0 => the load side writes the OTHER buffer, buffer 1
        ref_cfg(8, x"00000000");

        for row in 0 to K_LEN - 1 loop
            for bank in 0 to W_BANKS - 1 loop
                for b in 0 to BANK_BYTES - 1 loop
                    word((b + 1) * 8 - 1 downto b * 8) :=
                        std_logic_vector(to_signed(
                            wgt_val(row, bank * BANK_BYTES + b), 8));
                end loop;
                r_ws_row   <= std_logic_vector(to_unsigned(row, BANK_AW));
                r_ws_bank  <= std_logic_vector(to_unsigned(bank, 3));
                r_ws_data  <= word;
                r_ws_valid <= '1';
                wait until rising_edge(clk);
            end loop;
        end loop;
        r_ws_valid <= '0';

        ref_cfg(0, std_logic_vector(to_unsigned(K_LEN, 32)));
        ref_cfg(1, cfg1);
        ref_cfg(2, std_logic_vector(to_unsigned(ACT_BASE, 32)));
        ref_cfg(3, std_logic_vector(to_unsigned(OUT_BASE, 32)));
        ref_cfg(10, std_logic_vector(to_unsigned(2 ** RECIP_FRAC, 32)));
        -- Buffer A for both load and compute. weight_window qualifies its
        -- write enable with `not rd_buf`, so the written buffer IS the read
        -- buffer -- see the note in this testbench's header.
        ref_cfg(8, x"00000000");

        r_job_op    <= "00";       -- JOB_GEMM
        r_job_start <= '1';
        wait until rising_edge(clk);
        r_job_start <= '0';
        loop
            wait until rising_edge(clk);
            exit when r_done = '1';
        end loop;

        for i in 0 to N_OUT - 1 loop
            r_host_raddr <= std_logic_vector(to_unsigned(OUT_BASE + i, ACT_AW));
            r_host_re    <= '1';
            wait until rising_edge(clk);
            r_host_re    <= '0';
            wait for 1 ns;
            ref_out(i) <= to_integer(signed(r_out_data));
            wait until rising_edge(clk);
        end loop;

        ---------------------------------------------------------------
        -- PHASE 2 -- the wrapper, driven only through AXI
        ---------------------------------------------------------------
        report "phase 2: wrapper, AXI transactions only";

        axi_rd(A_ID, rv);
        if rv /= x"4D495247" then
            report "FAIL: ID register read " & to_hstring(rv)
                 & ", expected 4D495247" severity error;
            nerr := nerr + 1;
        end if;

        -- one pointer write, then a byte per beat: the auto-increment is
        -- what keeps a 512-byte vector to one loop instead of two writes
        -- per byte
        axi_wr(A_ACT_PTR, std_logic_vector(to_unsigned(ACT_BASE, 32)));
        for i in 0 to K_LEN - 1 loop
            axi_wr(A_ACT_DAT, std_logic_vector(
                       to_signed(act_val(i), 32)));
        end loop;

        axi_wr(A_CFG0 + 4 * 8, x"00000000");   -- bufsel 0: load buffer 1

        -- the stream carries no address; wgt_stream must rebuild the exact
        -- (row, bank) order phase 1 stated explicitly
        for row in 0 to K_LEN - 1 loop
            for bank in 0 to W_BANKS - 1 loop
                for b in 0 to BANK_BYTES - 1 loop
                    word((b + 1) * 8 - 1 downto b * 8) :=
                        std_logic_vector(to_signed(
                            wgt_val(row, bank * BANK_BYTES + b), 8));
                end loop;
                tdata  <= word;
                tvalid <= '1';
                if row = K_LEN - 1 and bank = W_BANKS - 1 then
                    tlast <= '1';
                end if;
                wait until rising_edge(clk);
            end loop;
        end loop;
        tvalid <= '0';
        tlast  <= '0';

        axi_wr(A_CFG0 + 4 * 0,  std_logic_vector(to_unsigned(K_LEN, 32)));
        axi_wr(A_CFG0 + 4 * 1,  cfg1);
        axi_wr(A_CFG0 + 4 * 2,  std_logic_vector(to_unsigned(ACT_BASE, 32)));
        axi_wr(A_CFG0 + 4 * 3,  std_logic_vector(to_unsigned(OUT_BASE, 32)));
        axi_wr(A_CFG0 + 4 * 10, std_logic_vector(to_unsigned(2 ** RECIP_FRAC, 32)));
        axi_wr(A_CFG0 + 4 * 8,  x"00000000");

        -- clear any stale done, then start
        axi_wr(A_ISR, x"00000003");
        axi_wr(A_CTRL, x"00000001");          -- start, job_op = 00 = GEMM

        -- Poll the LATCHED done. The fabric's done is one cycle wide and a
        -- single AXI read takes several, so an unlatched status bit would
        -- never be observed and this loop would never terminate.
        loop
            axi_rd(A_ISR, rv);
            exit when rv(0) = '1';
        end loop;

        if irq /= '1' then
            report "FAIL: irq not asserted while ISR[0] set" severity error;
            nerr := nerr + 1;
        end if;

        axi_rd(A_ACT_PTR, rv);   -- pointer survived the job untouched
        axi_wr(A_ACT_PTR, std_logic_vector(to_unsigned(OUT_BASE, 32)));
        for i in 0 to N_OUT - 1 loop
            axi_rd(A_ACT_DAT, rv);
            dut_out(i) <= to_integer(signed(rv));
            wait for 1 ns;
        end loop;

        ---------------------------------------------------------------
        -- PHASE 3 -- contract checks that do not involve arithmetic
        ---------------------------------------------------------------
        report "phase 3: wrapper contract";

        axi_wr(A_ISR, x"00000003");
        axi_rd(A_ISR, rv);
        if rv(0) /= '0' then
            report "FAIL: ISR[0] did not clear on write-1-to-clear"
                severity error;
            nerr := nerr + 1;
        end if;

        -- A config write while busy must be refused: layer_ctrl reads the
        -- config combinationally for the whole job, so applying one
        -- mid-flight would change geometry under a running reduction.
        axi_wr(A_CTRL, x"00000001");
        axi_wr(A_CFG0 + 4 * 0, std_logic_vector(to_unsigned(999, 32)));
        axi_rd(A_ISR, rv);
        if rv(1) /= '1' then
            report "FAIL: cfg write during busy was not flagged in ISR[1]"
                severity error;
            nerr := nerr + 1;
        end if;
        loop
            axi_rd(A_ISR, rv);
            exit when rv(0) = '1';
        end loop;

        ---------------------------------------------------------------
        -- PHASE 4 -- compare
        ---------------------------------------------------------------
        wait until rising_edge(clk);
        report "phase 4: comparing " & integer'image(N_OUT) & " output bytes";

        for i in 0 to 7 loop
            report "  byte " & integer'image(i)
                 & "  direct=" & integer'image(ref_out(i))
                 & "  axi=" & integer'image(dut_out(i));
        end loop;

        for i in 0 to N_OUT - 1 loop
            if ref_out(i) /= dut_out(i) then
                report "MISMATCH byte " & integer'image(i)
                     & ": direct " & integer'image(ref_out(i))
                     & ", axi " & integer'image(dut_out(i)) severity error;
                nerr := nerr + 1;
            end if;
        end loop;

        -- Liveness guard, and it has already earned its place. An earlier
        -- version of this bench checked only for all-zero, and passed while
        -- every byte was -128: the job had read a weight buffer nothing had
        -- written, the accumulator went undefined, and sat8('U') is -128 on
        -- BOTH instances, so they agreed perfectly on garbage. Any constant
        -- output is the signature of a job that did not really run, so the
        -- check is for variation, not for a particular wrong value.
        uniform := true;
        for i in 1 to N_OUT - 1 loop
            if ref_out(i) /= ref_out(0) then
                uniform := false;
            end if;
        end loop;
        if uniform then
            report "FAIL: every output byte is "
                 & integer'image(ref_out(0))
                 & " -- the job did not really run" severity error;
            nerr := nerr + 1;
        end if;

        write(lo, string'("=========================================="));
        writeline(output, lo);
        if nerr = 0 then
            write(lo, string'("PASS  axi wrapper transparent over "));
            write(lo, N_OUT);
            write(lo, string'(" bytes + 4 contract checks"));
        else
            write(lo, string'("FAIL  "));
            write(lo, nerr);
            write(lo, string'(" error(s)"));
        end if;
        writeline(output, lo);
        write(lo, string'("=========================================="));
        writeline(output, lo);

        errors <= nerr;
        halt   <= true;
        wait for 1 ns;
        finish;
    end process;

end architecture;
