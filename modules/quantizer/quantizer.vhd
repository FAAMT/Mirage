library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all; -- Assuming the array types are defined here

entity quantizer is
    generic
    (
        NUM_QUANTIZERS  : positive := TILE_H * TILE_W; -- Number of parallel quantizers
        INPUT_WIDTH     : positive := QKV_BRAM_DATA_W; -- Width of input data
        OUTPUT_WIDTH    : positive := MM_IN_DATA_W;    -- Width of output data
        SHIFT_AMOUNT    : natural  := MM_IN_DATA_W;    -- Number of bits to shift
        SHIFT_DIRECTION : string   := "RIGHT"          -- "LEFT" or "RIGHT"
    );
    port
    (
        -- Input data array (60 elements of 16 bits each)
        data_in : in quantizer_input_array;
        -- Output data array (60 elements of 8 bits each)
        data_out : out quantizer_output_array
    );
end quantizer;

architecture Behavioral of quantizer is
    -- Function to perform the shift operation
    function apply_shift(input : signed; amount : natural; direction : string) return signed is
    begin
        if direction = "LEFT" then
            return shift_left(input, amount);
        else
            return shift_right(input, amount);
        end if;
    end function;
begin
    -- Combinational quantization process
    process (data_in)
        variable shifted_data : signed(INPUT_WIDTH - 1 downto 0);
    begin
        for i in 0 to NUM_QUANTIZERS - 1 loop
            -- Apply shift and resize directly on array elements
            shifted_data := apply_shift(signed(data_in(i)), SHIFT_AMOUNT, SHIFT_DIRECTION);

            -- Assign to output array
            data_out(i) <= std_logic_vector(resize(shifted_data, OUTPUT_WIDTH));
        end loop;
    end process;
end Behavioral;