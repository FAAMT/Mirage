-- =====================================================================
-- addr_math_tb -- self-checking test for the address arithmetic in
-- mirage_v2_pkg.addr_mul, at the exact parameters layer_ctrl uses.
--
-- WHY THIS EXISTS
--
-- On 2026-07-30 Vivado 2022.2 rejected layer_ctrl with
--   ERROR: [Synth 8-690] width mismatch; target has 11 bits, source has 44
-- and behind that error sat a SILENT bug of the same origin: numeric_std's
-- "*"(UNSIGNED, NATURAL) converts the natural to the LEFT operand's width,
-- so `head * HEAD_DIM` with a 4-bit head is TO_UNSIGNED(64,4) = 0 and the
-- product is 0 for every head. Every head would have addressed head 0's
-- K/V. Neither GHDL's analyse/elaborate nor Vivado's synthesis objects to
-- that -- only running it does.
--
-- So this checks VALUES, not just widths, and checks the property that
-- actually matters for a memory address generator: the map is INJECTIVE.
-- An aliasing score address silently corrupts attention instead of failing.
--
--   ghdl -a --std=08 rtl_v2/pkg/mirage_v2_pkg.vhd
--   ghdl -a --std=08 rtl_v2/pkg/addr_math_tb.vhd
--   ghdl -e --std=08 addr_math_tb && ghdl -r --std=08 addr_math_tb
-- =====================================================================
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library work;
use work.mirage_v2_pkg.all;

entity addr_math_tb is
end entity;

architecture tb of addr_math_tb is

    -- Mirrors layer_ctrl's declared widths exactly. If those change, this
    -- test must be updated deliberately rather than drifting.
    constant HEAD_W : positive := 4;   -- signal head   : unsigned(3 downto 0)
    constant TOK_W  : positive := 5;   -- signal sm_tok : unsigned(4 downto 0)
    constant COL_W  : positive := 5;   -- signal sm_col : unsigned(4 downto 0)

    signal errors : natural := 0;

begin

    check : process
        variable head  : unsigned(HEAD_W - 1 downto 0);
        variable tok   : unsigned(TOK_W - 1 downto 0);
        variable col   : unsigned(COL_W - 1 downto 0);
        variable a     : unsigned(SCORE_AW - 1 downto 0);
        variable base  : unsigned(ACT_AW - 1 downto 0);
        variable seen  : integer_vector(0 to 2 ** SCORE_AW - 1);
        variable errs  : natural := 0;
        variable addr  : natural;
    begin
        ---------------------------------------------------------------
        -- 1. head * HEAD_DIM, the silent one. Expect 0,64,...,448.
        ---------------------------------------------------------------
        for h in 0 to NUM_HEADS - 1 loop
            head := to_unsigned(h, HEAD_W);
            base := addr_mul(head, HEAD_DIM, ACT_AW);
            if to_integer(base) /= h * HEAD_DIM then
                report "FAIL head base: head=" & integer'image(h)
                     & " got " & integer'image(to_integer(base))
                     & " want " & integer'image(h * HEAD_DIM)
                    severity error;
                errs := errs + 1;
            end if;
        end loop;

        -- The bug this replaces would have produced 0 for every head, so
        -- assert the range is actually distinct, not merely equal to a
        -- formula that could also collapse.
        if NUM_HEADS > 1 then
            assert to_integer(addr_mul(to_unsigned(1, HEAD_W), HEAD_DIM, ACT_AW))
                   /= to_integer(addr_mul(to_unsigned(0, HEAD_W), HEAD_DIM, ACT_AW))
                report "FAIL head base collapsed to a constant"
                severity error;
        end if;

        ---------------------------------------------------------------
        -- 2. Score address: head-major, then query row, then key col.
        --    Must be injective over the whole reachable domain and must
        --    stay inside the 2**SCORE_AW address space.
        ---------------------------------------------------------------
        seen := (others => 0);
        for h in 0 to NUM_HEADS - 1 loop
            for t in 0 to MAX_TOK - 1 loop
                for c in 0 to MAX_TOK - 1 loop
                    head := to_unsigned(h, HEAD_W);
                    tok  := to_unsigned(t, TOK_W);
                    col  := to_unsigned(c, COL_W);

                    a := addr_mul(head, MAX_TOK * MAX_TOK, SCORE_AW)
                         + addr_mul(tok, MAX_TOK, SCORE_AW)
                         + resize(col, SCORE_AW);
                    addr := to_integer(a);

                    if addr /= (h * MAX_TOK + t) * MAX_TOK + c then
                        report "FAIL score addr: h=" & integer'image(h)
                             & " t=" & integer'image(t)
                             & " c=" & integer'image(c)
                             & " got " & integer'image(addr)
                            severity error;
                        errs := errs + 1;
                    end if;

                    seen(addr) := seen(addr) + 1;
                end loop;
            end loop;
        end loop;

        -- Injectivity: every reachable address hit exactly once.
        for i in 0 to NUM_HEADS * MAX_TOK * MAX_TOK - 1 loop
            if seen(i) /= 1 then
                report "FAIL score addr aliasing at " & integer'image(i)
                     & " hit " & integer'image(seen(i)) & " times"
                    severity error;
                errs := errs + 1;
            end if;
        end loop;

        -- And the region must exactly fill the address space it was sized
        -- for -- neither overflowing it nor wasting a BRAM on slack.
        assert NUM_HEADS * MAX_TOK * MAX_TOK = 2 ** SCORE_AW
            report "score region " & integer'image(NUM_HEADS * MAX_TOK * MAX_TOK)
                 & " /= address space " & integer'image(2 ** SCORE_AW)
                 & " (not fatal, but the budget assumed a snug fit)"
            severity warning;

        ---------------------------------------------------------------
        if errs = 0 then
            report "PASS: head bases 0.." & integer'image((NUM_HEADS - 1) * HEAD_DIM)
                 & " correct, and "
                 & integer'image(NUM_HEADS * MAX_TOK * MAX_TOK)
                 & " score addresses are injective and in range"
                severity note;
        else
            report "FAILED with " & integer'image(errs) & " error(s)"
                severity failure;
        end if;

        errors <= errs;
        wait;
    end process;

end architecture;
