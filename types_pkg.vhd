library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package types_pkg is

    -- Constants
    constant DATA_WIDTH : integer := 8;
    constant TILE_H     : integer := 12;
    constant TILE_W     : integer := 5;

    -- Useful Types
    type signed_vector is array (natural range <>) of signed(DATA_WIDTH - 1 downto 0);
    type mac_array is array (natural range <>, natural range <>) of signed(8 - 1 downto 0);
    type accum_array is array (natural range <>) of signed(31 downto 0);

    -- Flatten signed_vector to std_logic_vector
    -- Function prototypes (only declaration here)
    function flatten_signed_vector(vec : signed_vector) return std_logic_vector;
    function flatten_mac_array(mat     : mac_array) return std_logic_vector;

end package types_pkg;