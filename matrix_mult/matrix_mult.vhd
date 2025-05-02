-- 12x5 DSP-Mapped Matrix Multiply Core (INT8)
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity matrix_mult is
  generic
  (
    DATA_WIDTH : integer := 8;
    TILE_H     : integer := 12;
    TILE_W     : integer := 5
  );
  port
  (
    clk      : in std_logic;
    rst      : in std_logic;
    start    : in std_logic;
    clear    : in std_logic;
    input_a  : in std_logic_vector(DATA_WIDTH * TILE_W - 1 downto 0); -- 5 INT8 inputs
    input_b  : in mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);        -- 12x5 INT8 weights
    output_d : out std_logic_vector(32 * TILE_H - 1 downto 0);        -- 12 INT32 outputs
    done     : out std_logic
  );
end entity;

architecture rtl of matrix_mult is

  type accum_array is array(0 to TILE_H - 1) of signed(31 downto 0);
  type signed_vector is array (natural range <>) of signed(DATA_WIDTH - 1 downto 0);

  signal a_vec     : signed_vector(0 to TILE_W - 1);
  signal w_mat     : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
  signal accum     : accum_array := (others => (others => '0'));
  signal accum_val : accum_array := (others => (others => '0'));
  signal computing : std_logic   := '0';
  signal done_i    : std_logic   := '0';

begin

  done <= done_i;

  -- Load Input & Weight Matrix
  process (input_a, input_b)
  begin
    w_mat <= input_b;

    for j in 0 to TILE_W - 1 loop
      a_vec(j) <= signed(input_a((j + 1) * DATA_WIDTH - 1 downto j * DATA_WIDTH));
    end loop;
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
        elsif start = '1' and computing = '0' then
        computing <= '1';
        done_i    <= '1';

        elsif computing = '1' then
        for i in 0 to TILE_H - 1 loop
          accum(i) <= accum(i) + accum_val(i);
        end loop;

        computing <= '0';
        done_i    <= '0';
        else
        done_i <= '0';
      end if;
    end if;
  end process;

  -- Accumulator Logic
  -- This process computes all 12 dot product
  -- outputs for 5 elements in parallel.
  accum_proc : process (computing)
    variable tmp_val : signed(31 downto 0);
  begin
    if computing = '1' then
      for i in 0 to TILE_H - 1 loop
        tmp_val := (others => '0');
        for j in 0 to TILE_W - 1 loop
          tmp_val := tmp_val + resize(w_mat(i, j) * a_vec(j), 32);
        end loop;
        accum_val(i) <= tmp_val;
      end loop;
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