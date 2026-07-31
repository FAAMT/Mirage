-- =====================================================================
-- wgt_stream.vhd -- AXI4-Stream sink for the weight window
--
-- The weight path is the bandwidth-critical direction: a layer loads
-- 1,048,576 window words, which at 32 bits per cycle and 100 MHz is
-- 10.5 ms against roughly 232,000 cycles of compute. The design is
-- weight-bound about 4.5 : 1, so this path -- not the DSP count -- sets
-- real throughput. It gets a streaming port and a DMA, while everything
-- else makes do with PIO over AXI4-Lite.
--
-- Addressing is generated here rather than carried on the bus. A window
-- row is N_LANES = 32 bytes, which is W_BANKS = 8 bank words of 32 bits,
-- so a linear stream maps exactly:
--
--     word i  ->  bank = i mod 8,  row = i / 8
--
-- That is a counter. Sending the address alongside the data would mean
-- TUSER, which an AXI DMA cannot generate from a plain memory buffer,
-- and would triple the bus width for information the sink already knows.
--
-- TLAST returns the counter to zero, so one descriptor is one window and
-- a short transfer cannot leave the next one skewed by a partial row.
--
-- No backpressure: weight_window has a write strobe and no ready, and it
-- accepts a word every cycle unconditionally, so TREADY is tied high
-- while enabled. Holding it high is not a shortcut -- the write port has
-- no arbitration to lose against, because reads come from the OTHER
-- buffer by construction.
-- =====================================================================
library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity wgt_stream is
    port
    (
        clk     : in std_logic;
        rst     : in std_logic;     -- active high, as the rest of the design

        -- AXI4-Stream slave
        s_axis_tvalid : in  std_logic;
        s_axis_tready : out std_logic;
        s_axis_tdata  : in  std_logic_vector(BANK_W - 1 downto 0);
        s_axis_tlast  : in  std_logic;

        -- weight_window load port
        wstream_valid : out std_logic;
        wstream_data  : out std_logic_vector(BANK_W - 1 downto 0);
        wstream_row   : out std_logic_vector(BANK_AW - 1 downto 0);
        wstream_bank  : out std_logic_vector(2 downto 0);

        -- observability: beats taken since the last TLAST
        beat_count : out std_logic_vector(31 downto 0)
    );
end entity;

architecture rtl of wgt_stream is
    signal row   : unsigned(BANK_AW - 1 downto 0) := (others => '0');
    signal bank  : unsigned(2 downto 0)           := (others => '0');
    signal beats : unsigned(31 downto 0)          := (others => '0');
    signal fire  : std_logic;
begin

    s_axis_tready <= '1';
    fire          <= s_axis_tvalid;   -- tready is constant '1'

    beat_count <= std_logic_vector(beats);

    -- The address is presented with the data it belongs to, both
    -- registered on the same edge. This is the alignment class that
    -- produced eight of the thirteen Layer C defects, so the strobe, the
    -- data and the address are written in one place and never in two.
    process (clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                row           <= (others => '0');
                bank          <= (others => '0');
                beats         <= (others => '0');
                wstream_valid <= '0';
                wstream_data  <= (others => '0');
                wstream_row   <= (others => '0');
                wstream_bank  <= (others => '0');
            else
                wstream_valid <= '0';

                if fire = '1' then
                    wstream_valid <= '1';
                    wstream_data  <= s_axis_tdata;
                    wstream_row   <= std_logic_vector(row);
                    wstream_bank  <= std_logic_vector(bank);
                    beats         <= beats + 1;

                    if s_axis_tlast = '1' then
                        row   <= (others => '0');
                        bank  <= (others => '0');
                        beats <= (others => '0');
                    elsif bank = W_BANKS - 1 then
                        bank <= (others => '0');
                        row  <= row + 1;     -- wraps at BANK_DEPTH by width
                    else
                        bank <= bank + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;

end architecture;
