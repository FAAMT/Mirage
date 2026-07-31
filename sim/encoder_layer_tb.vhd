-- =====================================================================
-- encoder_layer_tb -- a command interpreter, so the PS model can drive
-- the real RTL.
--
-- WHY A COMMAND INTERPRETER RATHER THAN A HAND-WRITTEN TESTBENCH
--
-- The thing that has to be verified is not "does a matmul work" -- the
-- unit testbenches cover that -- but "does the whole op chain, driven the
-- way the PS will actually drive it, reproduce a specific transformer
-- layer to the bit". That means the stimulus IS the PS's job schedule,
-- and the schedule depends on real weights and real token embeddings.
-- Encoding that in VHDL literals would mean regenerating the testbench
-- every time a scale or a token changes.
--
-- So this reads a command script and writes a result file. model_ref/
-- generates the script from the MarianMT weights, runs GHDL, reads the
-- results back, and compares them against the bit-exact int8 model. The
-- RTL sees exactly the sequence a Zynq PS would issue over AXI.
--
-- COMMANDS (one per line, whitespace separated, # starts a comment)
--
--   RST                       assert reset for a few cycles
--   CFG  <addr> <data>        write a job config register  (hex)
--   ACT  <addr> <byte>        write one activation byte    (hex)
--   WIN  <row> <bank> <word>  write one weight-window word (hex)
--   JOB  <op>                 pulse job_start, wait for done
--   DUMP <addr> <count>       read activation bytes, one per line, to out
--   DUMPS <addr> <count>      read score words, one per line, to out
--   MARK <text>               echo a label into the result file
--   END                       stop
--
-- Paths are generics so several runs can share one build.
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use STD.ENV.all;
use IEEE.STD_LOGIC_TEXTIO.all;

library work;
use work.mirage_v2_pkg.all;

entity encoder_layer_tb is
    generic (
        SCRIPT : string := "cmd.txt";
        RESULT : string := "res.txt"
    );
end entity;

architecture tb of encoder_layer_tb is

    constant TCK : time := 10 ns;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal job_start : std_logic := '0';
    signal job_op    : std_logic_vector(1 downto 0) := (others => '0');
    signal busy, done : std_logic;

    signal cfg_we   : std_logic := '0';
    signal cfg_addr : std_logic_vector(3 downto 0) := (others => '0');
    signal cfg_data : std_logic_vector(31 downto 0) := (others => '0');

    signal host_we    : std_logic := '0';
    signal host_waddr : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');
    signal host_wdata : std_logic_vector(ACT_W - 1 downto 0) := (others => '0');
    signal host_re    : std_logic := '0';
    signal host_raddr : std_logic_vector(ACT_AW - 1 downto 0) := (others => '0');

    signal ws_valid : std_logic := '0';
    signal ws_data  : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');
    signal ws_row   : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal ws_bank  : std_logic_vector(2 downto 0) := (others => '0');
    signal ws_buf   : std_logic;

    signal out_valid : std_logic;
    signal out_data  : std_logic_vector(ACT_W - 1 downto 0);

    signal hsc_re    : std_logic := '0';
    signal hsc_raddr : std_logic_vector(SCORE_AW - 1 downto 0) := (others => '0');
    signal hsc_dout  : std_logic_vector(SCORE_W - 1 downto 0);

    -- ---------------------------------------------------------------
    -- text helpers
    -- ---------------------------------------------------------------
    function hexval(c : character) return integer is
    begin
        case c is
            when '0' to '9' => return character'pos(c) - character'pos('0');
            when 'a' to 'f' => return character'pos(c) - character'pos('a') + 10;
            when 'A' to 'F' => return character'pos(c) - character'pos('A') + 10;
            when others     => return -1;
        end case;
    end function;

    -- read a whitespace-delimited token
    procedure get_tok(variable l : inout line; variable t : out string;
                      variable n : out natural) is
        variable c  : character;
        variable ok : boolean;
        variable i  : natural := 0;
    begin
        t := (t'range => ' ');
        -- skip leading space
        loop
            if l = null or l'length = 0 then
                n := 0;
                return;
            end if;
            read(l, c, ok);
            exit when not ok;
            if c /= ' ' and c /= HT then
                i    := 1;
                t(1) := c;
                exit;
            end if;
        end loop;
        if i = 0 then
            n := 0;
            return;
        end if;
        loop
            if l = null or l'length = 0 then exit; end if;
            read(l, c, ok);
            exit when not ok;
            if c = ' ' or c = HT then exit; end if;
            i    := i + 1;
            t(i) := c;
        end loop;
        n := i;
    end procedure;

    function to_int(s : string; n : natural) return integer is
        variable v : integer := 0;
        variable d : integer;
    begin
        for i in 1 to n loop
            d := hexval(s(i));
            exit when d < 0;
            v := v * 16 + d;
        end loop;
        return v;
    end function;

    -- Hex straight to a bit vector. Going via `integer` overflows on any
    -- word with bit 31 set, and a weight row is full 32-bit data.
    function to_slv(s : string; n : natural; w : natural)
        return std_logic_vector is
        variable r : std_logic_vector(w - 1 downto 0) := (others => '0');
        variable d : integer;
    begin
        for i in 1 to n loop
            d := hexval(s(i));
            exit when d < 0;
            r := r(w - 5 downto 0) & std_logic_vector(to_unsigned(d, 4));
        end loop;
        return r;
    end function;

    function dec_int(s : string; n : natural) return integer is
        variable v : integer := 0;
    begin
        for i in 1 to n loop
            exit when s(i) < '0' or s(i) > '9';
            v := v * 10 + (character'pos(s(i)) - character'pos('0'));
        end loop;
        return v;
    end function;

begin

    clk <= not clk after TCK / 2;

    dut : entity work.encoder_layer
        port map (clk => clk, rst => rst,
                  job_start => job_start, job_op => job_op,
                  busy => busy, done => done,
                  cfg_we => cfg_we, cfg_addr => cfg_addr, cfg_data => cfg_data,
                  host_we => host_we, host_waddr => host_waddr,
                  host_wdata => host_wdata,
                  host_re => host_re, host_raddr => host_raddr,
                  wstream_valid => ws_valid, wstream_data => ws_data,
                  wstream_row => ws_row, wstream_bank => ws_bank,
                  wstream_buf => ws_buf,
                  out_valid => out_valid, out_data => out_data,
                  host_sc_re => hsc_re, host_sc_raddr => hsc_raddr,
                  host_sc_dout => hsc_dout);

    ---------------------------------------------------------------------
    interp : process
        file fin  : text;
        file fout : text;
        variable l, lo : line;
        variable tok   : string(1 to 64);
        variable n     : natural;
        variable a, b, c : integer;
        variable njob  : natural := 0;
    begin
        file_open(fin, SCRIPT, read_mode);
        file_open(fout, RESULT, write_mode);

        rst <= '1';
        for i in 0 to 7 loop wait until rising_edge(clk); end loop;
        rst <= '0';
        wait until rising_edge(clk);

        while not endfile(fin) loop
            readline(fin, l);
            get_tok(l, tok, n);
            next when n = 0;
            next when tok(1) = '#';

            if tok(1 to n) = "RST" then
                rst <= '1';
                for i in 0 to 7 loop wait until rising_edge(clk); end loop;
                rst <= '0';
                wait until rising_edge(clk);

            elsif tok(1 to n) = "CFG" then
                get_tok(l, tok, n); a := to_int(tok, n);
                cfg_addr <= std_logic_vector(to_unsigned(a, 4));
                get_tok(l, tok, n);
                cfg_data <= to_slv(tok, n, 32);
                cfg_we   <= '1';
                wait until rising_edge(clk);
                cfg_we   <= '0';

            elsif tok(1 to n) = "ACT" then
                get_tok(l, tok, n); a := to_int(tok, n);
                get_tok(l, tok, n); b := to_int(tok, n);
                host_waddr <= std_logic_vector(to_unsigned(a, ACT_AW));
                host_wdata <= std_logic_vector(to_unsigned(b, ACT_W));
                host_we    <= '1';
                wait until rising_edge(clk);
                host_we    <= '0';

            elsif tok(1 to n) = "WIN" then
                get_tok(l, tok, n); a := to_int(tok, n);
                get_tok(l, tok, n); b := to_int(tok, n);
                ws_row   <= std_logic_vector(to_unsigned(a, BANK_AW));
                ws_bank  <= std_logic_vector(to_unsigned(b, 3));
                get_tok(l, tok, n);
                ws_data  <= to_slv(tok, n, BANK_W);
                ws_valid <= '1';
                wait until rising_edge(clk);
                ws_valid <= '0';

            elsif tok(1 to n) = "WROW" then
                -- One whole weight-window row: eight bank words on one
                -- line. A layer loads ~786,000 bank words, so writing them
                -- one per line made the script file eight times larger and
                -- textio parsing the dominant cost of the whole run.
                get_tok(l, tok, n); a := to_int(tok, n);
                for bk in 0 to W_BANKS - 1 loop
                    get_tok(l, tok, n);
                    ws_row   <= std_logic_vector(to_unsigned(a, BANK_AW));
                    ws_bank  <= std_logic_vector(to_unsigned(bk, 3));
                    ws_data  <= to_slv(tok, n, BANK_W);
                    ws_valid <= '1';
                    wait until rising_edge(clk);
                end loop;
                ws_valid <= '0';

            elsif tok(1 to n) = "JOB" then
                get_tok(l, tok, n); a := to_int(tok, n);
                job_op    <= std_logic_vector(to_unsigned(a, 2));
                job_start <= '1';
                wait until rising_edge(clk);
                job_start <= '0';
                wait until rising_edge(clk);
                -- a job may take hundreds of thousands of cycles
                while done = '0' loop
                    wait until rising_edge(clk);
                end loop;
                njob := njob + 1;

            elsif tok(1 to n) = "DUMP" then
                get_tok(l, tok, n); a := to_int(tok, n);
                get_tok(l, tok, n); b := dec_int(tok, n);
                for i in 0 to b - 1 loop
                    host_raddr <= std_logic_vector(to_unsigned(a + i, ACT_AW));
                    host_re    <= '1';
                    wait until rising_edge(clk);
                    host_re    <= '0';
                    wait for 1 ns;
                    write(lo, to_integer(signed(out_data)));
                    writeline(fout, lo);
                    wait until rising_edge(clk);
                end loop;

            elsif tok(1 to n) = "DUMPS" then
                get_tok(l, tok, n); a := to_int(tok, n);
                get_tok(l, tok, n); b := dec_int(tok, n);
                for i in 0 to b - 1 loop
                    hsc_raddr <= std_logic_vector(to_unsigned(a + i, SCORE_AW));
                    hsc_re    <= '1';
                    wait until rising_edge(clk);
                    hsc_re    <= '0';
                    wait for 1 ns;
                    write(lo, to_integer(signed(hsc_dout)));
                    writeline(fout, lo);
                    wait until rising_edge(clk);
                end loop;

            elsif tok(1 to n) = "MARK" then
                write(lo, string'("MARK "));
                get_tok(l, tok, n);
                write(lo, tok(1 to n));
                writeline(fout, lo);

            elsif tok(1 to n) = "END" then
                exit;
            else
                report "unknown command: " & tok(1 to n) severity warning;
            end if;
        end loop;

        write(lo, string'("DONE "));
        write(lo, njob);
        writeline(fout, lo);
        file_close(fout);
        file_close(fin);
        report "interpreter finished, " & integer'image(njob) & " jobs"
            severity note;
        -- Stop here rather than let the free-running clock burn through
        -- --stop-time after the script is exhausted: a 96 us script left
        -- GHDL simulating an idle design for the remaining 2 seconds.
        finish;
    end process;

end architecture;
