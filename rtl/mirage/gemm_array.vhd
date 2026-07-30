-- 12x5 DSP-Mapped Matrix Multiply Core (INT8)
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity gemm_array is
  generic
  (
    DATA_W : integer := 8;
    TILE_H : integer := 12;
    TILE_W : integer := 5
  );
  port
  (
    clk      : in std_logic;
    rst      : in std_logic;
    start    : in std_logic;
    clear    : in std_logic;
    input_a  : in int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1); -- 12x5 INT8 inputs
    input_b  : in int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1); -- 12x5 INT8 weights
    output_d : out std_logic_vector(32 * TILE_H - 1 downto 0);       -- 12 INT32 outputs
    done     : out std_logic
  );
end entity;

architecture rtl of gemm_array is

  type accum_array is array(0 to TILE_H - 1) of signed(31 downto 0);
  type signed_vector is array (natural range <>) of signed(DATA_W - 1 downto 0);

  signal a_vec     : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
  signal w_mat     : int8_gemm_array(0 to TILE_H - 1, 0 to TILE_W - 1);
  signal accum     : accum_array := (others => (others => '0'));
  signal accum_val : accum_array := (others => (others => '0'));
  signal computing : std_logic   := '0';
  signal done_i    : std_logic   := '0';

begin

  done <= done_i;

  -- Load Input & Weight Matrix
  process (input_a, input_b)
  begin

    a_vec <= input_a;
    w_mat <= input_b;

  end process;

  -- Matrix Mult Logic
  process (clk, rst)
  begin
    if rst = '1' then
      done_i    <= '0';
      computing <= '0';
      accum     <= (others => (others => '0'));

    elsif rising_edge(clk) then
      if clear = '1' then
        for i in 0 to TILE_H - 1 loop
          accum(i) <= (others => '0');
        end loop;
      else
        if computing = '1' then
          for i in 0 to TILE_H - 1 loop
            accum(i) <= accum(i) + accum_val(i);
          end loop;
        end if;
      end if;

      if start = '1' and computing = '0' then

        computing <= '1';
        done_i    <= '1';

      elsif computing = '1' or start = '1' then
        computing <= '0';
        done_i    <= '0';
      else
        computing <= start;
        done_i    <= '0';
      end if;
    end if;
  end process;

  -- Accumulator Logic
  -- This process computes all 12 dot product
  -- outputs for 5 elements in parallel.
  --
  -- CLOCKED 2026-07-30. Was a combinational process:
  --     process (input_a, input_b, start)   -- note: read `computing`, which
  --                                         -- was NOT in the sensitivity list
  --     -- if rising_edge(clk) then         -- commented out by the author
  --
  -- Two defects, one consequence:
  --   1. `if computing = '1'` with no else, in a combinational process, made
  --      accum_val a transparent LATCH -- Vivado 2022.2 emitted 12x
  --      [Synth 8-327] "inferring latch for accum_val_reg[n]" and 240 LD cells.
  --   2. The 60 int8 multiplies were therefore combinational logic feeding a
  --      latch. DSP48E1 inference requires the multiply to feed a clocked
  --      register, so Vivado mapped ALL of it to fabric: 0 DSP, ~6,818 LUT,
  --      1,056 CARRY4 -- against reference/w11-resource-map.md's estimate of
  --      60 DSP and "LUT/FF comfortably under 30%".
  --
  -- Clocking it is what the commented-out line intended, removes the latch,
  -- and puts a register where DSP inference wants one.
  --
  -- COSTS ONE PIPELINE CYCLE: accum_val is now registered, so the accumulate
  -- in the main process sees the previous cycle's product sum. The latched
  -- version had no such delay -- but it was also a race between latch
  -- propagation and the clock edge, which is why it must not stay.
  --
  -- Original preserved at reference/gemm_array.latched-accum.vhd
  accum_proc : process (clk)
    variable tmp_val : signed(31 downto 0);
  begin
    if rising_edge(clk) then
      if computing = '1' then
        for i in 0 to TILE_H - 1 loop
          tmp_val := (others => '0');
          for j in 0 to TILE_W - 1 loop
            tmp_val := tmp_val + resize(input_b(i, j) * input_a(i, j), 32);
          end loop;
          accum_val(i) <= tmp_val;
        end loop;
      end if;
    end if;
  end process;

  -- Output Logic
  output_proc : process (accum)
  begin
    for i in 0 to TILE_H - 1 loop
      output_d((i + 1) * 32 - 1 downto i * 32) <= std_logic_vector(accum(i));
    end loop;
  end process;

end architecture;