library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package types_pkg is

    -- Constants
    constant DATA_WIDTH : integer := 8;
    constant TILE_H : integer := 12;
    constant TILE_W : integer := 5;

    -- Useful Types
    type signed_vector is array (natural range <>) of signed(DATA_WIDTH - 1 downto 0);
    type mac_array is array (natural range <>, natural range <>) of signed(DATA_WIDTH - 1 downto 0);
    type accum_array is array (natural range <>) of signed(31 downto 0);

end package types_pkg;