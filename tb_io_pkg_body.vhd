library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

package body tb_io_pkg is

    --------------------------------------------------------------------
    -- read_int8_matrix: generic 2D matrix reader
    --------------------------------------------------------------------
    procedure read_int8_matrix(
        file_path : in string;
        rows      : in natural;
        cols      : in natural;
        data_out  : out matrix_int8
    ) is
        file f                         : text open read_mode is file_path;
        variable L                     : line;
        variable token                 : string(1 to 32);
        variable c                     : natural;
        variable r                     : natural;
        variable val                   : integer;
        variable good                  : boolean;
        variable ch                    : character;
        variable token_valid           : boolean;
        variable token_len             : natural;
        variable debug_line_num        : natural := 0;
        variable debug_token_num       : natural := 0;
        variable first_token_processed : boolean := false;
        variable new_token             : boolean := true;
    begin
        report "DEBUG: Starting to parse file: " & file_path;
        report "DEBUG: Expected dimensions: " & integer'image(rows) & " rows x " & integer'image(cols) & " columns";

        r := 0;
        while (r < rows and not endfile(f)) loop
            readline(f, L);
            debug_line_num := debug_line_num + 1;
            new_token      := true;
            -- Skip blank or comment lines
            if (L'length = 0) then
                report "DEBUG: Skipping empty line";
                next;
            elsif (L'length >= 2 and L(1) = '-' and L(2) = '-') then
                report "DEBUG: Skipping comment line: " & L.all;
                next;
            end if;

            c               := 0;
            debug_token_num := 0;

            -- report "DEBUG: Starting to parse columns for row " & integer'image(r);

            while (c < cols and L'length > 0) loop
                -- Skip whitespace and commas
                while L'length > 0 loop
                    if L'length = 0 then
                        report "DEBUG: End of line reached while skipping whitespace";
                        exit;
                    end if;

                    read(L, ch, good);

                    if not good then
                        report "DEBUG: Failed to read character while skipping whitespace";
                        exit;
                    end if;

                    exit when (ch /= ' ' and ch /= ',' and ch /= HT and ch /= '(' and ch /= ')' and ch /= '=' and ch /= '>');
                end loop;

                if not good or L'length = 0 then
                    report "DEBUG: End of line or failed read, moved to next line";
                    exit;
                end if;

                -- Build token
                token       := (others => ' ');
                token_len   := 1;
                token_valid := false;

                if good then
                    token(1)    := ch;
                    token_valid := true;

                    -- report "Current Token => " & ch;

                    for i in 2 to token'length loop
                        if L'length = 0 then
                            exit;
                        end if;

                        read(L, ch, good);

                        if not good or ch = ' ' or ch = ',' or ch = HT or ch = '(' or ch = ')' or ch = '=' or ch = '>' then
                            exit;
                        end if;

                        token(i)  := ch;
                        token_len := i;
                    end loop;
                end if;

                debug_token_num := debug_token_num + 1;

                -- Only convert if token begins with digit or minus sign
                if token_valid and ((token(1) >= '0' and token(1) <= '9') or token(1) = '-') then
                    -- Safely handle value conversion
                    val := integer'value(token(1 to token_len));

                    -- Check if value fits in 8 bits with Q1.7 format (range -128 to 127)
                    if val <- 128 or val > 127 then
                        report "WARNING: Value " & integer'image(val) & " outside Q1.7 range at row " & integer'image(r) & ", col " & integer'image(c);
                    end if;

                    data_out(r, c) := to_signed(val, 8);
                    c              := c + 1;
                end if;
            end loop;

            -- Only increment row counter if we found some columns
            if c > 0 then
                r := r + 1;
            end if;
        end loop;

        if r < rows then
            report "WARNING: Expected " & integer'image(rows) & " rows but only found " & integer'image(r);
        end if;

        file_close(f);
    end procedure;

    --------------------------------------------------------------------
    -- read_token_embeddings: identical format, aliased for clarity
    --------------------------------------------------------------------
    procedure read_token_embeddings(
        file_path : in string;
        rows      : in natural;
        cols      : in natural;
        data_out  : out matrix_int8
    ) is
    begin
        -- just call the generic reader
        read_int8_matrix(file_path, rows, cols, data_out);
    end procedure;

end package body tb_io_pkg;