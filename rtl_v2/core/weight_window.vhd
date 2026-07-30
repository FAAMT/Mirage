-- =====================================================================
-- weight_window.vhd — double-buffered, bank-partitioned weight BRAM
--
-- Serves the MAC array N_LANES weights every cycle while the next window
-- streams in from DDR behind it. This is the "weight streaming, shared
-- buffer reuse" mitigation the MVP plan's risk register called for and
-- v1 never built -- v1 declared 464 KB of weights as resident signal
-- arrays on a device with 225 KB of BRAM, then read them 60-wide.
--
-- BANK PARTITIONING (the plan's W7 item, "prevent bank conflicts")
--
-- A weight row is N_LANES = 32 consecutive int8. It is split across
-- W_BANKS = 8 banks of BANK_BYTES = 4 bytes each:
--
--     bank b holds bytes [4b .. 4b+3] of every row
--     row k lives at address k in ALL banks
--
-- So one read at address k, issued to all 8 banks at once, returns the
-- whole 32-byte row. Every bank sees exactly ONE read address per cycle,
-- which is why this infers block RAM where v1's flat array could not.
-- There are no conflicts by construction: banks are indexed by byte
-- position, never by data value.
--
-- Each bank is BANK_W = 32 bits x BANK_DEPTH = 1024, which is precisely
-- one RAMB36E1 in 32-bit mode. Sizing to the primitive stops Vivado from
-- splitting or merging banks into something unintended.
--
-- DOUBLE BUFFERING
--
-- 2 buffers x 8 banks = 16 RAMB36 = 32% of the device's 50. `buf_sel`
-- picks the compute-side buffer; writes always land in the other one, so
-- weight load for tile n+1 overlaps compute on tile n. weight_stream_ctrl
-- owns buf_sel and flips it at tile boundaries.
--
-- Capacity: one buffer holds BANK_DEPTH rows x N_LANES cols = 1024 x 32
-- int8 = 32 KB, enough for a full K_STEPS = 512-deep reduction over a
-- 32-column output tile (16 KB) with room to spare.
-- =====================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

library work;
use work.mirage_v2_pkg.all;

entity weight_window is
    port
    (
        clk : in std_logic;

        -- compute side: one row of N_LANES weights per cycle
        rd_en   : in  std_logic;
        rd_row  : in  std_logic_vector(BANK_AW - 1 downto 0);
        rd_buf  : in  std_logic;                  -- which buffer to read
        w_bus   : out wgt_bus_t;

        -- load side: one 32-bit bank word per cycle, from the DDR stream.
        -- Writes go to the buffer NOT selected by rd_buf.
        wr_en   : in std_logic;
        wr_row  : in std_logic_vector(BANK_AW - 1 downto 0);
        wr_bank : in std_logic_vector(2 downto 0);   -- 0 .. W_BANKS-1
        wr_data : in std_logic_vector(BANK_W - 1 downto 0)
    );
end entity;

architecture rtl of weight_window is

    -- per-buffer, per-bank read data
    type bank_bus_t is array (0 to W_BANKS - 1) of std_logic_vector(BANK_W - 1 downto 0);
    signal rd_a, rd_b : bank_bus_t;

    signal rd_buf_q : std_logic := '0';

begin

    ---------------------------------------------------------------------
    -- rd_buf must be delayed by the BRAM read latency (1 cycle) so the
    -- output mux selects the buffer the address was issued to, not
    -- whatever is selected when the data arrives. Getting this wrong
    -- would silently mix buffers at every tile boundary.
    ---------------------------------------------------------------------
    sel_delay : process (clk)
    begin
        if rising_edge(clk) then
            rd_buf_q <= rd_buf;
        end if;
    end process;

    ---------------------------------------------------------------------
    -- Buffer A and buffer B, W_BANKS each. Write enable is qualified by
    -- the buffer NOT being read, so compute and load never touch the
    -- same primitive.
    ---------------------------------------------------------------------
    gen_banks : for b in 0 to W_BANKS - 1 generate
        signal we_a, we_b : std_logic;
    begin
        we_a <= wr_en and (not rd_buf) when unsigned(wr_bank) = b else '0';
        we_b <= wr_en and rd_buf       when unsigned(wr_bank) = b else '0';

        buf_a : entity work.bram_sdp
            generic map (DATA_W => BANK_W, ADDR_W => BANK_AW, DOUT_REG => false)
            port map (
                clk => clk,
                we => we_a, waddr => wr_row, din => wr_data,
                re => rd_en, raddr => rd_row, dout => rd_a(b)
            );

        buf_b : entity work.bram_sdp
            generic map (DATA_W => BANK_W, ADDR_W => BANK_AW, DOUT_REG => false)
            port map (
                clk => clk,
                we => we_b, waddr => wr_row, din => wr_data,
                re => rd_en, raddr => rd_row, dout => rd_b(b)
            );

        -- Bank b supplies lanes 4b .. 4b+3. Concatenating all banks in
        -- order reconstructs the N_LANES-wide row.
        w_bus((b + 1) * BANK_W - 1 downto b * BANK_W)
            <= rd_a(b) when rd_buf_q = '0' else rd_b(b);
    end generate;

end architecture;
