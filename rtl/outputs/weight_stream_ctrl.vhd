-- =============================================================================
--  File      : weight_stream_ctrl.vhd  (DRAFT — ghdl-analyzed, not unit-simmed)
--  Purpose   : Multi-layer scheduler / weight streaming controller. Sequences
--              NUM_LAYERS encoder layers through ONE shared datapath (time-
--              multiplexed — per the W6 resource profile, sequential reuse is
--              the only option that fits the 7z007s).
--  Schedule  : for l in 0..NUM_LAYERS-1:
--                1. stream layer l's weights from external memory into the
--                   inactive weight buffer (mem_* -> wr_*)
--                2. flip buf_sel and pulse layer_start
--                3. wait layer_done
--              Overlap of prefetch with compute (true ping-pong) is a future
--              optimization; the buffer select already supports it.
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

entity weight_stream_ctrl is
    generic
    (
        NUM_LAYERS   : positive := 3;      -- compile-time layer-count scaling (task risk note)
        WEIGHT_WORDS : positive := 278528; -- int8 words per layer (W7: proj+hl+ol+wq/wk/wv/wo)
        ADDR_W       : positive := 20
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start : in  std_logic;            -- begin full L-layer pass
        done  : out std_logic;            -- all layers complete
        busy  : out std_logic;

        -- Encoder-layer datapath handshake
        layer_start : out std_logic;
        layer_done  : in  std_logic;
        layer_index : out integer range 0 to NUM_LAYERS - 1;
        buf_sel     : out std_logic;      -- which weight buffer the datapath reads

        -- External weight memory stream (DDR-side DMA or host)
        mem_addr  : out unsigned(ADDR_W - 1 downto 0);
        mem_req   : out std_logic;
        mem_data  : in  std_logic_vector(7 downto 0);
        mem_valid : in  std_logic;

        -- Weight BRAM write port (into the INACTIVE buffer)
        wr_addr : out unsigned(ADDR_W - 1 downto 0);
        wr_data : out std_logic_vector(7 downto 0);
        wr_en   : out std_logic;
        wr_bank : out std_logic           -- buffer being written (always ~buf_sel)
    );
end entity;

architecture rtl of weight_stream_ctrl is
    type state_type is (idle, prefetch, launch, running, final);
    signal state    : state_type := idle;
    signal layer    : integer range 0 to NUM_LAYERS - 1 := 0;
    signal word_ctr : integer range 0 to WEIGHT_WORDS - 1 := 0;
    signal cur_buf  : std_logic := '0';

    -------------------------------------------------------------------
    -- Vivado VHDL-2008 workaround (added 2026-07-29)
    --
    -- This architecture reads its own 'out' ports. That is legal
    -- VHDL-2008 and GHDL accepts it, but Vivado's synthesiser does not
    -- implement it and errors with [Synth 8-10557]. Drive and read
    -- internally, mirror to the ports. Entity interface unchanged, so
    -- nothing that instantiates this module is affected.
    --
    -- Original preserved in reference/
    -------------------------------------------------------------------
    signal mem_addr_i : unsigned(ADDR_W - 1 downto 0);

begin

    mem_addr <= mem_addr_i;

    layer_index <= layer;
    buf_sel     <= cur_buf;
    wr_bank     <= not cur_buf;

    process (clk, rst)
    begin
        if rst = '1' then
            state       <= idle;
            layer       <= 0;
            word_ctr    <= 0;
            cur_buf     <= '0';
            done        <= '0';
            busy        <= '0';
            layer_start <= '0';
            mem_req     <= '0';
            wr_en       <= '0';

        elsif rising_edge(clk) then
            layer_start <= '0';
            wr_en       <= '0';
            done        <= '0';

            case state is

                when idle =>
                    if start = '1' then
                        busy     <= '1';
                        layer    <= 0;
                        cur_buf  <= '0';
                        mem_addr_i <= (others => '0');
                        word_ctr <= 0;
                        mem_req  <= '1';
                        state    <= prefetch;
                    end if;

                -- Stream WEIGHT_WORDS bytes for the current layer into the
                -- inactive buffer, one outstanding word at a time.
                when prefetch =>
                    if mem_valid = '1' then
                        wr_addr <= to_unsigned(word_ctr, ADDR_W);
                        wr_data <= mem_data;
                        wr_en   <= '1';
                        if word_ctr = WEIGHT_WORDS - 1 then
                            mem_req  <= '0';
                            word_ctr <= 0;
                            state    <= launch;
                        else
                            word_ctr <= word_ctr + 1;
                            mem_addr_i <= mem_addr_i + 1;
                        end if;
                    end if;

                -- Hand the freshly filled buffer to the datapath and go.
                when launch =>
                    layer_start <= '1';
                    cur_buf     <= not cur_buf;
                    state       <= running;

                when running =>
                    if layer_done = '1' then
                        if layer = NUM_LAYERS - 1 then
                            state <= final;
                        else
                            layer    <= layer + 1;
                            mem_addr_i <= (others => '0');
                            word_ctr <= 0;
                            mem_req  <= '1';
                            state    <= prefetch;
                        end if;
                    end if;

                when final =>
                    done  <= '1';
                    busy  <= '0';
                    state <= idle;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

end architecture;
