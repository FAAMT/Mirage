library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

package tb_io_pkg is

    -- 8-bit Q1.7 signed fixed-point matrix type
    type matrix_int8_t is array (natural range <>, natural range <>) of signed(7 downto 0);
    subtype matrix_int8 is matrix_int8_t;

    -- Read any 2-D matrix of 8-bit signed ints from a text file
    procedure read_int8_matrix(
        file_path : in string;
        rows      : in natural;
        cols      : in natural;
        data_out  : out matrix_int8
    );

    -- Read token embeddings: one vector of length D_MODEL per line
    procedure read_token_embeddings(
        file_path : in string;
        rows      : in natural;
        cols      : in natural;
        data_out  : out matrix_int8
    );

end package tb_io_pkg;