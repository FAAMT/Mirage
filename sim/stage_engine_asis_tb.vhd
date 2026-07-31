-- =====================================================================
-- stage_engine_asis_tb -- does stage_engine honour its OWN contract?
--
-- This testbench deliberately holds stage_engine to the weakest possible
-- standard: the one written in its own header comment.
--
--     xpose = '0'  (xV)   : addr = base + row*lanes  + lane
--     xpose = '1'  (QK^T) : addr = base + lane*rows  + row
--
-- and, from weight_window's header, "row k lives at address k in ALL
-- banks", with "bank b holds bytes [4b .. 4b+3] of every row".
--
-- Put together, after a staging pass the window must satisfy
--
--     window[r][j] = act[addr(r, j)]      for r < rows, j < lanes
--
-- Nothing here assumes the fix. If the module works as documented, this
-- passes. It is written before any change so that the failures it reports
-- are evidence about the code as received, not about the code as edited.
--
-- RESULT, on the module as received: 45 of 160 staged bytes wrong.
--
-- KEPT AS A RECORD, NOT AS A LIVE TEST. stage_engine has since gained
-- `stride` and `row_ofs` ports, so this file no longer analyses against it
-- and is deliberately absent from build.sh's list. To re-run it, check the
-- module out at commit ba8c15a. sim/stage_engine_tb.vhd is the live test.
--
--   ghdl -a --std=08 --workdir=work ... && ghdl -r --std=08 stage_engine_asis_tb
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity stage_engine_asis_tb is
end entity;

architecture tb of stage_engine_asis_tb is

    constant TCK : time := 10 ns;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal go, busy, fin : std_logic := '0';
    signal base  : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal rows  : std_logic_vector(6 downto 0) := (others => '0');
    signal lanes : std_logic_vector(6 downto 0) := (others => '0');
    signal xpose : std_logic := '0';

    signal act_re    : std_logic;
    signal act_raddr : std_logic_vector(ACT_AW - 1 downto 0);
    signal act_data  : act_t := (others => '0');

    signal w_we   : std_logic;
    signal w_row  : std_logic_vector(BANK_AW - 1 downto 0);
    signal w_bank : std_logic_vector(2 downto 0);
    signal w_data : std_logic_vector(BANK_W - 1 downto 0);

    signal rd_en  : std_logic := '0';
    signal rd_row : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal w_bus  : wgt_bus_t;

    -- Behavioural activation memory. act[a] = a mod 251, which is
    -- injective over any 251-address span, so a wrong address shows up as
    -- a wrong VALUE rather than coincidentally matching.
    function act_of(a : natural) return integer is
    begin
        return (a mod 251) - 125;   -- spread over int8, never 0-heavy
    end function;

    signal done : boolean := false;

begin

    clk <= not clk after TCK / 2;

    -- activation memory model: 1-cycle synchronous read, like bram_sdp
    act_model : process (clk)
    begin
        if rising_edge(clk) then
            if act_re = '1' then
                act_data <= to_signed(act_of(to_integer(unsigned(act_raddr))), ACT_W);
            end if;
        end if;
    end process;

    dut : entity work.stage_engine
        port map (clk => clk, rst => rst,
                  go => go, busy => busy, fin => fin,
                  base => base, rows => rows, lanes => lanes, xpose => xpose,
                  act_re => act_re, act_raddr => act_raddr, act_data => act_data,
                  w_we => w_we, w_row => w_row, w_bank => w_bank, w_data => w_data);

    win : entity work.weight_window
        port map (clk => clk,
                  rd_en => rd_en, rd_row => rd_row, rd_buf => '0', w_bus => w_bus,
                  wr_en => w_we, wr_row => w_row, wr_bank => w_bank, wr_data => w_data);

    ---------------------------------------------------------------------
    stim : process
        variable errs  : natural := 0;
        variable checked : natural := 0;
        variable shown : natural := 0;

        procedure run_pass(b : natural; r : natural; l : natural; xp : std_logic) is
        begin
            base  <= std_logic_vector(to_unsigned(b, ACT_AW));
            rows  <= std_logic_vector(to_unsigned(r, 7));
            lanes <= std_logic_vector(to_unsigned(l, 7));
            xpose <= xp;
            wait until rising_edge(clk);
            go <= '1';
            wait until rising_edge(clk);
            go <= '0';
            wait until fin = '1' and rising_edge(clk);
            -- let the final packed word land
            for i in 0 to 4 loop wait until rising_edge(clk); end loop;
        end procedure;

        procedure check_window(name : string; b : natural; r : natural;
                               l : natural; xp : std_logic) is
            variable want : integer;
            variable got  : integer;
            variable addr : natural;
        begin
            report "---- " & name & ": base=" & integer'image(b)
                 & " rows=" & integer'image(r) & " lanes=" & integer'image(l)
                 & " xpose=" & std_logic'image(xp) severity note;
            for rr in 0 to r - 1 loop
                rd_row <= std_logic_vector(to_unsigned(rr, BANK_AW));
                rd_en  <= '1';
                wait until rising_edge(clk);
                rd_en  <= '0';
                wait until rising_edge(clk);   -- data valid now
                for jj in 0 to l - 1 loop
                    if xp = '1' then
                        addr := b + jj * r + rr;
                    else
                        addr := b + rr * l + jj;
                    end if;
                    want := act_of(addr);
                    got  := to_integer(get_wgt(w_bus, jj));
                    checked := checked + 1;
                    if got /= want then
                        errs := errs + 1;
                        if shown < 12 then
                            shown := shown + 1;
                            report "  MISMATCH row=" & integer'image(rr)
                                 & " lane=" & integer'image(jj)
                                 & " addr=" & integer'image(addr)
                                 & " want=" & integer'image(want)
                                 & " got=" & integer'image(got)
                                severity warning;
                        end if;
                    end if;
                end loop;
            end loop;
        end procedure;
    begin
        rst <= '1';
        for i in 0 to 4 loop wait until rising_edge(clk); end loop;
        rst <= '0';
        wait until rising_edge(clk);

        -- Case 1: the xV geometry as layer_ctrl issues it, but with lanes
        -- cut to N_LANES so it at least FITS one window row. rows = n_tok.
        run_pass(16#6000#, 4, 32, '0');
        check_window("xV-like, rows=4 lanes=32", 16#6000#, 4, 32, '0');

        -- Case 2: the QK^T geometry, reduced so the run is quick.
        --         rows = 8 (a slice of HEAD_DIM), lanes = 4 (n_tok).
        run_pass(16#4000#, 8, 4, '1');
        check_window("QK^T-like, rows=8 lanes=4", 16#4000#, 8, 4, '1');

        report "==== checked " & integer'image(checked) & " window bytes, "
             & integer'image(errs) & " wrong" severity note;
        if errs = 0 then
            report "PASS" severity note;
        else
            report "FAIL: stage_engine does not honour its own documented "
                 & "address contract" severity failure;
        end if;
        done <= true;
        wait;
    end process;

end architecture;
