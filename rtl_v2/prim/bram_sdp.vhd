-- =====================================================================
-- bram_sdp.vhd — simple dual-port BRAM (1 write port, 1 read port)
--
-- THE canonical memory primitive for v2. Every memory in the design goes
-- through this entity or its true-dual-port sibling. Nothing declares a
-- bare array signal and indexes it, which is what v1 did everywhere and
-- why v1 inferred zero BRAM.
--
-- This is the Vivado-recommended SDP template (UG901, "RAM HDL Coding
-- Techniques"), deviated from in no respect. The four rules that matter:
--
--   1. SYNCHRONOUS READ. `dout` is assigned inside `rising_edge(clk)`.
--      An asynchronous read forces distributed RAM (LUTRAM) at best.
--
--   2. NO RESET ANYWHERE IN THIS PROCESS. Block RAM contents have no
--      reset in silicon, so a reset on `ram` -- or even on `dout` in the
--      same process -- makes the pattern unrecognisable and Vivado falls
--      back to registers. v1's arrays all had
--      `sig <= (others => (others => '0'))` in a reset branch; that alone
--      disqualified them.
--
--   3. EXACTLY ONE READ ADDRESS AND ONE WRITE ADDRESS. v1 read one array
--      at 60 different indices in a single process. BRAM offers 2 ports
--      and LUTRAM 16, so 60 collapsed to flip-flops plus a mux tree --
--      2,688 bits of storage cost 11,587 LUTs in qk_score_bram.
--
--   4. ram_style = "block" states the intent explicitly rather than
--      relying on Vivado's size heuristic.
--
-- Read latency is 1 cycle (2 with DOUT_REG, which buys Fmax at the cost
-- of one more cycle -- use it on the weight path where the sequencer can
-- absorb the delay).
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity bram_sdp is
    generic
    (
        DATA_W   : positive := 32;
        ADDR_W   : positive := 10;
        DOUT_REG : boolean  := false   -- extra output register for timing
    );
    port
    (
        clk : in std_logic;

        -- write port
        we    : in std_logic;
        waddr : in std_logic_vector(ADDR_W - 1 downto 0);
        din   : in std_logic_vector(DATA_W - 1 downto 0);

        -- read port
        re    : in std_logic;
        raddr : in std_logic_vector(ADDR_W - 1 downto 0);
        dout  : out std_logic_vector(DATA_W - 1 downto 0)
    );
end entity;

architecture rtl of bram_sdp is

    type ram_t is array (0 to 2 ** ADDR_W - 1) of std_logic_vector(DATA_W - 1 downto 0);
    signal ram : ram_t;

    -- Force block RAM. Without this Vivado applies a size heuristic and
    -- may choose distributed RAM for the smaller banks.
    attribute ram_style : string;
    attribute ram_style of ram : signal is "block";

    signal dout_i : std_logic_vector(DATA_W - 1 downto 0);
    signal dout_q : std_logic_vector(DATA_W - 1 downto 0);

begin

    -- No reset in this process. Deliberate -- see rule 2 in the header.
    mem_proc : process (clk)
    begin
        if rising_edge(clk) then
            if we = '1' then
                ram(to_integer(unsigned(waddr))) <= din;
            end if;
            if re = '1' then
                dout_i <= ram(to_integer(unsigned(raddr)));
            end if;
        end if;
    end process;

    opt_reg : if DOUT_REG generate
        reg_proc : process (clk)
        begin
            if rising_edge(clk) then
                dout_q <= dout_i;
            end if;
        end process;
        dout <= dout_q;
    end generate;

    no_reg : if not DOUT_REG generate
        dout <= dout_i;
    end generate;

end architecture;
