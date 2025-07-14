library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package types_pkg is
    -- Clk Period Constant
    -- := 200 Mhz
    constant clk_period : time := 5 ns;

    -- Model Dimension Constants
    constant NUM_HEADS   : integer := 8;
    constant HEAD_DIM    : integer := 64;
    constant MODEL_DIM   : integer := 512;
    constant TOKEN_DIM   : integer := 512;
    constant TOK_CONTEXT : integer := 4;

    -- Matrix Mult Constants
    constant TILE_H            : integer := 12;
    constant TILE_W            : integer := 5;
    constant MM_IN_DATA_WIDTH  : integer := 8;
    constant MM_OUT_DATA_WIDTH : integer := 32;

    -- MHA Weight Matrices Constants
    constant MHA_DATA_WIDTH : integer := 8;
    constant WEIGHT_SLICE   : integer := 512;
    constant Q_BASE         : integer := 0;
    constant K_BASE         : integer := Q_BASE + HEAD_DIM * WEIGHT_SLICE;
    constant V_BASE         : integer := K_BASE + HEAD_DIM * WEIGHT_SLICE;
    constant BASE_OFFSET    : integer := 3 * (HEAD_DIM * MODEL_DIM);

    -- QKV Cache Constants
    constant CACHE_DATA_WIDTH : integer := 16;
    constant QKV_CACHE_ELEM   : integer := 3 * TOK_CONTEXT * NUM_HEADS * HEAD_DIM;

    -- Type Declarations
    type signed_vector is array (natural range <>) of signed(MM_IN_DATA_WIDTH - 1 downto 0);
    type mac_array is array (natural range <>, natural range <>) of signed(MM_IN_DATA_WIDTH - 1 downto 0);
    type accum_array is array (natural range <>) of signed(MM_OUT_DATA_WIDTH - 1 downto 0);

    -- Function Declarations
    function flatten_signed_vector(vec : signed_vector) return std_logic_vector;
    function flatten_mac_array(mat     : mac_array) return std_logic_vector;

end package types_pkg;