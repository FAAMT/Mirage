library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

package body types_pkg is

    function flatten_signed_vector(vec : signed_vector) return std_logic_vector is
        variable res : std_logic_vector(vec'length * vec(0)'length - 1 downto 0);
    begin
        for i in vec'range loop
            res((i + 1) * vec(0)'length - 1 downto i * vec(0)'length) := std_logic_vector(vec(i));
        end loop;
        return res;
    end function;

    function flatten_mac_array(mat : mac_array) return std_logic_vector is
        variable res : std_logic_vector(mat'length(1) * mat'length(2) * mat(0, 0)'length - 1 downto 0);
        variable idx : integer := 0;
    begin
        for i in mat'range(1) loop
            for j in mat'range(2) loop
                res((idx + 1) * mat(0, 0)'length - 1 downto idx * mat(0, 0)'length) := std_logic_vector(mat(i, j));
                idx := idx + 1;
            end loop;
        end loop;
        return res;
    end function;

end package body;