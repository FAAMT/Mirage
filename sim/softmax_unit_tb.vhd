-- =====================================================================
-- softmax_unit_tb -- the exp stream and its sum, against real arithmetic
--
-- The reference is computed here from the same definition tools/
-- gen_exp_rom.py used to build the table:
--
--     idx    = clamp((max - s) >> EXP_ROM_SH, 0, 2^EXP_ROM_AW - 1)
--     exp_q  = round(exp(-idx / 256) * 2^16)          -- ROM contents
--     e8     = min(127, (exp_q * 127) >> 16)
--     sum8   = sum of e8 over the row
--
-- so a mismatch is a fault in the unit, not a disagreement about what exp
-- means.
--
-- WHAT THIS IS REALLY LOOKING FOR
--
-- Pipeline alignment. softmax_unit is address -> ROM -> requantise, and
-- every stage of that is a synchronous read or a register. If any stage
-- samples its input a cycle early the values come out SHIFTED: the row
-- still looks like a plausible softmax, the argmax still requantises to
-- something near 127, and the sum is still in range. Only an element-wise
-- comparison against a known row catches it.
--
-- The rows below are chosen so a shift cannot pass:
--   * the argmax sits in a different column in each row
--   * one row is strictly increasing, so a shift by one changes every term
--   * one row is flat, where e8 must be 127 everywhere and sum8 = n * 127
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use IEEE.MATH_REAL.all;

library work;
use work.mirage_v2_pkg.all;

entity softmax_unit_tb is
end entity;

architecture tb of softmax_unit_tb is

    constant TCK  : time    := 10 ns;
    constant NCOL : natural := 6;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal row_start : std_logic := '0';
    signal pass_exp  : std_logic := '0';
    signal s_valid   : std_logic := '0';
    signal s_data    : score_t   := (others => '0');

    signal e_valid : std_logic;
    signal e_data  : act_t;
    signal sum8    : std_logic_vector(SUM8_W - 1 downto 0);

    type row_t is array (0 to NCOL - 1) of integer;
    type e_t   is array (0 to NCOL - 1) of integer;

    -- the four rows under test
    type rows_t is array (0 to 3) of row_t;
    constant ROWS : rows_t := (
        ( 100,  -40,   17, -300,   62,  -11),   -- argmax at 0
        (-500, -120, -900,  333,  -75, -260),   -- argmax at 3
        (-300, -240, -180, -120,  -60,    0),   -- strictly increasing
        (  55,   55,   55,   55,   55,   55));  -- flat: every e8 = 127

    -- reference: the arithmetic the unit is supposed to implement
    function ref_e8(mx : integer; s : integer) return integer is
        variable mag : integer;
        variable idx : integer;
        variable eq  : integer;
    begin
        mag := mx - s;
        if mag < 0 then
            idx := 0;
        elsif mag > MAG_MAX then
            idx := 2 ** EXP_ROM_AW - 1;
        else
            idx := mag / (2 ** EXP_ROM_SH);
        end if;
        eq := integer(round(exp(-real(idx) / 256.0) * 65536.0));
        if eq > 65536 then
            eq := 65536;
        end if;
        return minimum(127, (eq * 127) / 65536);
    end function;

    signal got   : e_t := (others => 0);
    signal n_got : natural := 0;

begin

    clk <= not clk after TCK / 2;

    dut : entity work.softmax_unit
        port map (clk => clk, rst => rst,
                  row_start => row_start, pass_exp => pass_exp,
                  s_valid => s_valid, s_data => s_data,
                  e_valid => e_valid, e_data => e_data, sum8 => sum8);

    -- collect every exp value the unit emits
    collect : process (clk)
    begin
        if rising_edge(clk) then
            if row_start = '1' then
                n_got <= 0;
            elsif e_valid = '1' and n_got < NCOL then
                got(n_got) <= to_integer(e_data);
                n_got      <= n_got + 1;
            end if;
        end if;
    end process;

    stim : process
        variable errs : natural := 0;
        variable mx   : integer;
        variable want : e_t;
        variable wsum : integer;

        procedure drive_row(r : row_t; pex : std_logic) is
        begin
            pass_exp <= pex;
            for j in 0 to NCOL - 1 loop
                s_data  <= to_signed(r(j), SCORE_W);
                s_valid <= '1';
                wait until rising_edge(clk);
            end loop;
            s_valid <= '0';
            for i in 0 to 5 loop wait until rising_edge(clk); end loop;
        end procedure;
    begin
        rst <= '1';
        for i in 0 to 3 loop wait until rising_edge(clk); end loop;
        rst <= '0';
        wait until rising_edge(clk);

        for ri in 0 to 3 loop
            -- reference for this row
            mx := ROWS(ri)(0);
            for j in 1 to NCOL - 1 loop
                if ROWS(ri)(j) > mx then mx := ROWS(ri)(j); end if;
            end loop;
            wsum := 0;
            for j in 0 to NCOL - 1 loop
                want(j) := ref_e8(mx, ROWS(ri)(j));
                wsum    := wsum + want(j);
            end loop;

            -- pass 1: maximum. row_start seeds it.
            row_start <= '1';
            wait until rising_edge(clk);
            row_start <= '0';
            drive_row(ROWS(ri), '0');

            -- pass 2: exp. NO row_start -- the maximum must survive.
            drive_row(ROWS(ri), '1');

            report "row " & integer'image(ri) & ": max=" & integer'image(mx)
                 & " emitted " & integer'image(n_got) & " of "
                 & integer'image(NCOL) severity note;

            if n_got /= NCOL then
                errs := errs + 1;
                report "  WRONG COUNT: got " & integer'image(n_got)
                     & " exp values, expected " & integer'image(NCOL)
                    severity warning;
            end if;
            for j in 0 to NCOL - 1 loop
                if got(j) /= want(j) then
                    errs := errs + 1;
                    report "  col " & integer'image(j)
                         & " s=" & integer'image(ROWS(ri)(j))
                         & " want e8=" & integer'image(want(j))
                         & " got=" & integer'image(got(j)) severity warning;
                end if;
            end loop;
            if to_integer(unsigned(sum8)) /= wsum then
                errs := errs + 1;
                report "  sum8 want=" & integer'image(wsum)
                     & " got=" & integer'image(to_integer(unsigned(sum8)))
                    severity warning;
            end if;
        end loop;

        if errs = 0 then
            report "PASS: softmax_unit matches the reference on all four rows"
                severity note;
        else
            report integer'image(errs) & " softmax mismatches" severity failure;
        end if;
        wait;
    end process;

end architecture;
