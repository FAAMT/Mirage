library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;

entity softmax_tb is
end entity;

architecture tb of softmax_tb is
    -- Constant & Type Declarations
    constant DATA_WIDTH : integer := 16;

    type fetch_mac_array is array (0 to 191) of integer;
    signal mac_q : fetch_mac_array := (
        3200, -25216, 9344, -24192, 5248, -160, -9088, 12672,
        - 33664, 15744, -35712, 0, 4096, -8192, 8192, -16384,
        12288, -20480, 16384, -24576, 20480, -28672, 24576, -32768,
        28672, -36864, 32768, -40960, 36864, -45056, 40960, -49152,
        45056, -53248, 49152, -57344, 53248, -61440, 57344, -65536,
        61440, -81920, 65536, -65536, 69632, -81920, 73728, -87040,
        77824, -92160, 81920, -97280, 86016, -102400, 90112, -107520,
        94208, -112640, 98304, -117760, 102400, -122880, 106496, -128000,

        3200, -25216, 9344, -24192, 5248, -160, -9088, 12672,
        - 33664, 15744, -35712, 0, 4096, -8192, 8192, -16384,
        12288, -20480, 16384, -24576, 20480, -28672, 24576, -32768,
        28672, -36864, 32768, -40960, 36864, -45056, 40960, -49152,
        45056, -53248, 49152, -57344, 53248, -61440, 57344, -65536,
        61440, -81920, 65536, -65536, 69632, -81920, 73728, -87040,
        77824, -92160, 81920, -97280, 86016, -102400, 90112, -107520,
        94208, -112640, 98304, -117760, 102400, -122880, 106496, -128000,

        3200, -25216, 9344, -24192, 5248, -160, -9088, 12672,
        - 33664, 15744, -35712, 0, 4096, -8192, 8192, -16384,
        12288, -20480, 16384, -24576, 20480, -28672, 24576, -32768,
        28672, -36864, 32768, -40960, 36864, -45056, 40960, -49152,
        45056, -53248, 49152, -57344, 53248, -61440, 57344, -65536,
        61440, -81920, 65536, -65536, 69632, -81920, 73728, -87040,
        77824, -92160, 81920, -97280, 86016, -102400, 90112, -107520,
        94208, -112640, 98304, -117760, 102400, -122880, 106496, -128000,
        others => 0
    );

    -- CLK % RST
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    -- Control Logic
    signal start    : std_logic := '0';
    signal clear    : std_logic := '0';
    signal done     : std_logic;
    signal find_max : std_logic := '0';

    -- MAC Signals
    signal start_mac : std_logic := '0';
    signal clear_mac : std_logic := '0';
    signal fetch_mac : std_logic := '0';
    signal done_mac  : std_logic;

    -- Attention Signals
    signal fetch_attn : std_logic := '0';

    signal input_a  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => to_signed(0, 8)));
    signal input_b  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1) := (others => (others => to_signed(0, 8)));
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0); -- 12x INT32 outputs

    signal mac_q_init   : fetch_mac_array := mac_q;
    signal mac_in       : std_logic_vector(12 * 16 - 1 downto 0);
    signal mac_addr_off : integer := 0;

    -- Text Output Signals
    signal attn_pass : integer := 0;

begin
    -- Instantiate the softmax core
    uut : entity work.softmax
        generic
        map
        (
        DATA_WIDTH => 16
        )
        port map
        (
            clk => clk,
            rst => rst,

            start => start,
            done  => done,

            fetch_qkv  => fetch_mac,
            fetch_attn => fetch_attn,
            data_in    => mac_in,
            data_out   => open,

            -- Matrix Mult Interface
            start_mac => start_mac,
            clear_mac => clear_mac,
            done_mac  => done_mac,
            input_a   => input_a,
            input_b   => input_b,
            output_d  => output_d
        );

    -- Instantiate Matrix Multiply Core (MAC Array)
    mac_core : entity work.matrix_mult
        generic
        map (
        DATA_WIDTH => 8,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_mac,
        clear    => clear_mac,
        input_a  => input_a,
        input_b  => input_b,
        output_d => output_d, -- output not needed yet for this phase
        done     => done_mac
        );

    -- Clock process
    clk_process : process
    begin
        loop
            clk <= '0';
            wait for 2.5 ns;
            clk <= '1';
            wait for 2.5 ns;
        end loop;
    end process;

    -- Stimulus
    stim_proc : process
        variable idx : integer := 0;
    begin
        -- Reset
        rst <= '1';
        wait for 10 ns;
        rst <= '0';

        start <= '1';
        wait for 5 ns;
        start <= '0';

        mac_in <= (others => '0');

        wait for 50 ns;

        -- Fetch MAC Values
        for proj_iter in 0 to 3 loop
            mac_addr_off <= 0;
            mac_q        <= mac_q_init;
            wait for 5 ns;

            for i in 0 to 17 loop
                fetch_mac <= '1';

                for j in 0 to 11 loop
                    -- Quantize MAC Outputs 32 bit -> 16 bit
                    if (i < 5) then
                        mac_q(12 * i + j + mac_addr_off) <=
                        mac_q(12 * i + j + mac_addr_off) / 4;

                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                        std_logic_vector(to_signed(((mac_q(12 * i + j + mac_addr_off)) / 4) + 1000 * proj_iter, DATA_WIDTH));
                        elsif (i = 5) then
                        if j < 4 then
                            mac_q(12 * i + j + mac_addr_off) <=
                            mac_q(12 * i + j + mac_addr_off) / 4;

                            mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                            std_logic_vector(to_signed(((mac_q(12 * i + j + mac_addr_off)) / 4) + 1000 * proj_iter, DATA_WIDTH));
                        end if;
                        elsif (i < 11) then
                        mac_q(12 * i + j + mac_addr_off) <=
                        (mac_q(12 * i + j + mac_addr_off) / 8);

                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                        std_logic_vector(to_signed(100 * proj_iter + (mac_q(12 * i + j + mac_addr_off) / 8), DATA_WIDTH));
                        elsif (i = 11) then
                        if j < 4 then
                            mac_q(12 * i + j + mac_addr_off) <=
                            (mac_q(12 * i + j + mac_addr_off) / 8);

                            mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                            std_logic_vector(to_signed(100 * proj_iter + (mac_q(12 * i + j + mac_addr_off) / 8), DATA_WIDTH));

                        end if;
                        elsif (i < 17) then
                        mac_q(12 * i + j + mac_addr_off) <=
                        mac_q(12 * i + j + mac_addr_off) / 16;

                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                        std_logic_vector(to_signed(0 * proj_iter + mac_q(12 * i + j + mac_addr_off) / 16, DATA_WIDTH));
                        elsif (i = 17) then
                        if j < 4 then
                            mac_q(12 * i + j + mac_addr_off) <=
                            mac_q(12 * i + j + mac_addr_off) / 16;

                            mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <=
                            std_logic_vector(to_signed(0 * proj_iter + mac_q(12 * i + j + mac_addr_off) / 16, DATA_WIDTH));

                        end if;
                    end if;

                    -- MAC Addr Offset Logic
                    if ((12 * i + j + mac_addr_off) > 127) then
                        mac_addr_off <= - 16;

                        elsif ((12 * i + j + mac_addr_off) > 63) then
                        mac_addr_off <= - 8;
                        else
                        mac_addr_off <= 0;
                    end if;
                end loop;

                wait for 5 ns;
                fetch_mac <= '0';
                wait for 0.75 us;
            end loop;
        end loop;
        wait;
    end process;

    -------------------------------------------------------------------------------
    --  Attention-output capture
    --  • 12×32-bit words arrive in output_d whenever fetch_attn = '1'
    --  • Write them in triples so the file shows 4 × 64 groups:
    --      T0[ 0]  v0 v1 v2
    --      T0[ 1]  v3 v4 v5
    --      ...
    --      T3[63]  ...
    -------------------------------------------------------------------------------
    attn_capture : process (clk)
        type attention_array is array (0 to 63) of integer;
        type attention_arrays is array (0 to 3) of attention_array;

        variable attn_arrays : attention_arrays;
        variable array_idx   : integer range 0 to 3  := 0;
        variable element_idx : integer range 0 to 63 := 0;
        variable L           : line;
        file output_file     : text open write_mode is "attention_output.txt";
    begin
        if rising_edge(clk) then
            if fetch_attn = '1' then
                -- Initialize file header on first pass
                if attn_pass = 0 then
                    write(L, string'("=== Attention Output Arrays ==="));
                    writeline(output_file, L);
                end if;

                -- Update pass counter
                if attn_pass <= 21 then
                    attn_pass    <= attn_pass + 1;
                    else
                    attn_pass <= 0;
                end if;

                -- Process each row and append to current array
                for row in 0 to 11 loop
                    -- Only process elements divisible by 3 on last pass
                    if (attn_pass < 21) then
                        -- Increment element counter
                        element_idx := (row mod 3) + 3 * attn_pass;
                        array_idx   := row / 3;

                        -- Store value in current array
                        attn_arrays(array_idx)(element_idx) :=
                        to_integer(signed(output_d((row + 1) * 32 - 1 downto row * 32)));

                        report
                        "Row: " & integer'image(row) & " " &
                        "Pass: " & integer'image(attn_pass) & " " &
                        "Element: " & integer'image(element_idx) & " " &
                        "Array: " & integer'image(array_idx) & " " &
                        "Val: " & integer'image(attn_arrays(array_idx)(element_idx));

                        else
                        if ((row = 0) or (row = 3) or (row = 6) or (row = 9)) then
                            -- Only process elements divisible by 3 on last pass                                -- Increment element counter
                            element_idx := (row mod 3) + 3 * attn_pass;
                            array_idx   := row / 3;

                            -- Store value in current array
                            attn_arrays(array_idx)(element_idx) :=
                            to_integer(signed(output_d((row + 1) * 32 - 1 downto row * 32)));

                            report
                            "Row: " & integer'image(row) & " " &
                            "Pass: " & integer'image(attn_pass) & " " &
                            "Element: " & integer'image(element_idx) & " " &
                            "Array: " & integer'image(array_idx) & " " &
                            "Val: " & integer'image(attn_arrays(array_idx)(element_idx));
                        end if;
                    end if;
                end loop;

                -- When array is full, move to next array
                if attn_pass = 21 then
                    array_idx := 0;

                    for array_iter in 0 to 3 loop
                        writeline(output_file, L);

                        -- Write completed array to file
                        write(L, string'("Array_") & integer'image(array_idx) & string'(":"));
                        writeline(output_file, L);

                        for i in 0 to 63 loop
                            write(L, attn_arrays(array_idx)(i));
                            if i < 63 then
                                write(L, ' ');
                            end if;
                            if (i + 1) mod 3 = 0 then -- New line every 16 elements
                                writeline(output_file, L);
                            end if;
                        end loop;

                        array_idx := array_idx + 1;

                        -- Add spacing between arrays
                        writeline(output_file, L);
                    end loop;
                end if;
            end if;
        end if;
    end process;
end architecture;