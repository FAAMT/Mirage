library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity softmax_tb is
end entity;

architecture tb of softmax_tb is
    constant DATA_WIDTH : integer := 16;

    type mac_array is array (0 to 191) of integer;
    signal mac_q : mac_array := (
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

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    signal start     : std_logic := '0';
    signal done      : std_logic;
    signal fetch_mac : std_logic := '0';

    signal mac_in       : std_logic_vector(12 * DATA_WIDTH - 1 downto 0);
    signal mac_addr_off : integer := 0;
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

            fetch_qkv => fetch_mac,
            data_in   => mac_in,
            data_out  => open
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
        for i in 0 to 17 loop
            fetch_mac <= '1';

            for j in 0 to 11 loop
                -- Quantize MAC Outputs 32 bit -> 16 bit
                if (i = 5) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off) / 4;
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off) / 4, DATA_WIDTH));
                    end if;

                    elsif (i = 11) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off) / 4;
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off) / 4, DATA_WIDTH));
                    end if;

                    elsif (i = 17) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off) / 4;
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off) / 4, DATA_WIDTH));
                    end if;
                    else
                    mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off) / 4;
                    mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off) / 4, DATA_WIDTH));
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
            -- wait for 5 ns;

            wait for 0.75 us;
        end loop;

        wait for 5 us;

        start <= '1';
        wait for 5 ns;
        start <= '0';

        mac_in       <= (others => '0');
        mac_addr_off <= 0;

        wait for 50 ns;

        -- Fetch MAC Values
        for i in 0 to 17 loop
            fetch_mac <= '1';

            for j in 0 to 11 loop
                -- Quantize MAC Outputs 32 bit -> 16 bit
                if (i = 5) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off);
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off), DATA_WIDTH));
                    end if;

                    elsif (i = 11) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off);
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off), DATA_WIDTH));
                    end if;

                    elsif (i = 17) then
                    if j < 4 then
                        mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off);
                        mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off), DATA_WIDTH));
                    end if;
                    else
                    mac_q(12 * i + j + mac_addr_off)                           <= mac_q(12 * i + j + mac_addr_off);
                    mac_in((DATA_WIDTH * (j + 1) - 1) downto (DATA_WIDTH * j)) <= std_logic_vector(to_signed(mac_q(12 * i + j + mac_addr_off), DATA_WIDTH));
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
            -- wait for 5 ns;

            wait for 0.75 us;
        end loop;

        wait;
    end process;

end architecture;