-- rc2_hardening.vhd — DRAFT reference snippets (outputs/), RC2 checklist items
-- from the W17 robustness matrix. NOT wired into Mirage/ (read-only); these
-- are the reference implementations for the integration wrapper.
--
-- Item 1 (W17 §2): saturating int32 accumulator readback. The gemm accumulator
-- can overflow int32 at the 2048-wide FFN hidden layer under adversarial
-- all-max data (analytic worst case ~4x int32). Practically unreachable (LN
-- outputs are unit-scale), but saturating readback is zero-cost insurance.
--
-- Item 2 (W17 §4): defensive clamp at the range-reduced exp input. exp_q9_rr's
-- documented domain is z <= 0 (softmax scores are max-subtracted); z > 0 is
-- out-of-contract and unbounded. Clamp positive z to 0 at the input.
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package rc2_hardening_pkg is

    -- Item 1: saturating readback of a wider accumulator into int32.
    -- ACC_W = physical accumulator width (>= 33 to hold overflow headroom).
    function sat32(acc : signed) return signed;

    -- Item 2: defensive domain clamp for the exp input (Q9 domain).
    function clamp_exp_input(z : signed) return signed;

end package;

package body rc2_hardening_pkg is

    function sat32(acc : signed) return signed is
        constant MAX32 : signed(31 downto 0) := x"7FFFFFFF";
        constant MIN32 : signed(31 downto 0) := x"80000000";
    begin
        if acc'length <= 32 then
            return resize(acc, 32);
        elsif acc > resize(MAX32, acc'length) then
            return MAX32;
        elsif acc < resize(MIN32, acc'length) then
            return MIN32;
        else
            return acc(31 downto 0);
        end if;
    end function;

    function clamp_exp_input(z : signed) return signed is
    begin
        if z > 0 then
            return (z'range => '0');  -- z > 0 is out of contract: clamp to e^0
        end if;
        return z;
    end function;

end package body;
