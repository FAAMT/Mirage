-- ============================================================================
--  File      : qkv_cache.vhd
--  Author    : Fahad Tajiki
--  Date      : 10/07/2025
--  Purpose   : Implements an on-chip cache for Query, Key, and Value (QKV) vectors
--              in a Transformer Encoder Layer on FPGA. Stores three 1x64 wide 
--              vectors for up to 4 tokens and 8 heads, totaling 6144 int16 words
--              (approximately 12.3 KiB at 16-bit width).
-- ============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity qkv_cache is
    generic
    (
        -- Width of data elements in cache
        DATA_WIDTH : integer := CACHE_DATA_WIDTH
    );
    port
    (
        -- Clock and reset signals
        clk : in std_logic;
        rst : in std_logic;
        -- Control signal to initiate QKV fetch
        fetch_qkv : in std_logic;
        -- Cache read control signal
        cache_rd : in std_logic;
        -- Cache address input
        cache_addr : in std_logic;
        -- Input data vector for QKV storage
        data_in : in std_logic_vector(TILE_H * DATA_WIDTH - 1 downto 0);
        -- Output data vector from cache
        data_out : out std_logic_vector(TILE_H * DATA_WIDTH - 1 downto 0)
    );
end entity;

architecture rtl of qkv_cache is
    -- State machine states for QKV fetch sequence
    type state_type is (idle, fetch_q, fetch_k, fetch_v);
    -- Array type for QKV cache storage (6144 int16 words)
    type bram_qkv_type is array(0 to QKV_CACHE_ELEM - 1) of std_logic_vector(15 downto 0);

    -- Current and previous state registers
    signal state : state_type := idle;
    -- Counter for tracking tokens (0 to 3)
    signal tok_count : integer range 0 to TOK_CONTEXT - 1 := 0;
    -- Counter for tracking heads (0 to 7)
    signal set_ctr : integer range 0 to NUM_HEADS - 1 := 0;
    -- Counter for fetch passes (6 passes for HEAD_DIM / TILE_W)
    signal fetch_ctr : integer range 0 to 5 := 0;
    -- Block RAM array for QKV data
    signal bram_qkv : bram_qkv_type;
    -- Address for accessing QKV cache
    signal bram_qkv_addr : integer range 0 to QKV_CACHE_ELEM - 1 := 0;

begin
    -- Address calculation for QKV cache access
    process (state, fetch_ctr, set_ctr)
    begin
        case state is
            when fetch_q =>
                -- Calculate address for Query vector
                bram_qkv_addr <= 0 + 12 * fetch_ctr + 192 * set_ctr + 1536 * tok_count;
            when fetch_k =>
                -- Calculate address for Key vector
                bram_qkv_addr <= 64 + 12 * fetch_ctr + 192 * set_ctr + 1536 * tok_count;
            when fetch_v =>
                -- Calculate address for Value vector
                bram_qkv_addr <= 128 + 12 * fetch_ctr + 192 * set_ctr + 1536 * tok_count;
            when others =>
                -- Default to zero address in idle state
                bram_qkv_addr <= 0;
        end case;
    end process;

    -- Main control and data storage process
    process (clk, rst)
    begin
        if rst = '1' then
            -- Reset all control and data signals
            state     <= idle;
            tok_count <= 0;
            set_ctr   <= 0;
            fetch_ctr <= 0;
            bram_qkv  <= (others => (others => '0'));
            data_out  <= (others => '0');
        elsif rising_edge(clk) then
            case state is
                when idle =>
                    state <= fetch_q;

                when fetch_q =>
                    if fetch_qkv = '1' then
                        if fetch_ctr = 5 then
                            -- Last fetch pass for Query, move to Key
                            fetch_ctr <= 0;
                            state     <= fetch_k;
                            -- Store final 4 data elements
                            for i in 0 to 3 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        else
                            -- Store 12 data elements and increment fetch counter
                            fetch_ctr <= fetch_ctr + 1;
                            for i in 0 to 11 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        end if;
                    end if;

                when fetch_k =>
                    if fetch_qkv = '1' then
                        if fetch_ctr = 5 then
                            -- Last fetch pass for Key, move to Value
                            fetch_ctr <= 0;
                            state     <= fetch_v;
                            -- Store final 4 data elements
                            for i in 0 to 3 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        else
                            -- Store 12 data elements and increment fetch counter
                            fetch_ctr <= fetch_ctr + 1;
                            for i in 0 to 11 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        end if;
                    end if;

                when fetch_v =>
                    if fetch_qkv = '1' then
                        if fetch_ctr = 5 then
                            -- Last fetch pass for Value, complete QKV head
                            fetch_ctr <= 0;
                            state     <= idle;
                            if set_ctr = 7 then
                                -- All heads complete, reset head counter
                                set_ctr <= 0;
                                if tok_count = 3 then
                                    -- All tokens complete, reset token counter
                                    tok_count <= 0;
                                else
                                    -- Move to next token
                                    tok_count <= tok_count + 1;
                                end if;
                            else
                                -- Move to next head
                                set_ctr <= set_ctr + 1;
                            end if;
                            -- Store final 4 data elements
                            for i in 0 to 3 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        else
                            -- Store 12 data elements and increment fetch counter
                            fetch_ctr <= fetch_ctr + 1;
                            for i in 0 to 11 loop
                                bram_qkv(bram_qkv_addr + i) <= data_in((DATA_WIDTH * (i + 1) - 1) downto (DATA_WIDTH * i));
                            end loop;
                        end if;
                    end if;
            end case;
        end if;
    end process;

end architecture;