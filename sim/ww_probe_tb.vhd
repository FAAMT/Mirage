-- Minimal probe: is weight_window's first read after an idle gap stale?
-- Writes one known word, then reads the same row three times in a row.
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
library work;
use work.mirage_v2_pkg.all;

entity ww_probe_tb is
end entity;

architecture tb of ww_probe_tb is
    signal clk : std_logic := '0';
    signal rd_en : std_logic := '0';
    signal rd_row : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal w_bus : wgt_bus_t;
    signal wr_en : std_logic := '0';
    signal wr_row : std_logic_vector(BANK_AW - 1 downto 0) := (others => '0');
    signal wr_bank : std_logic_vector(2 downto 0) := (others => '0');
    signal wr_data : std_logic_vector(BANK_W - 1 downto 0) := (others => '0');
begin
    clk <= not clk after 5 ns;

    win : entity work.weight_window
        port map (clk => clk, rd_en => rd_en, rd_row => rd_row, rd_buf => '0',
                  w_bus => w_bus, wr_en => wr_en, wr_row => wr_row,
                  wr_bank => wr_bank, wr_data => wr_data);

    p : process
        procedure rd(tag : string) is
        begin
            rd_en <= '1';
            wait until rising_edge(clk);
            rd_en <= '0';
            wait for 1 ns;
            report tag & " lane0=" & integer'image(to_integer(get_wgt(w_bus, 0)))
                 & " lane1=" & integer'image(to_integer(get_wgt(w_bus, 1)))
                severity note;
        end procedure;
    begin
        -- write row 0, bank 0 = bytes {11, 22, 33, 44}
        wait until rising_edge(clk);
        wr_row  <= (others => '0');
        wr_bank <= "000";
        wr_data <= x"2C21160B";     -- 44,33,22,11 -> lane0=11 lane1=22
        wr_en   <= '1';
        wait until rising_edge(clk);
        wr_en   <= '0';
        wait until rising_edge(clk);

        -- an idle gap, exactly like the one between a staging pass and the
        -- checking loop that follows it
        for i in 0 to 5 loop wait until rising_edge(clk); end loop;

        rd("read 1 (first after idle)");
        rd("read 2");
        rd("read 3");
        wait;
    end process;
end architecture;
