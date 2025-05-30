library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.types_pkg.all;
use work.tb_io_pkg.all;

entity system_tb is
end entity;

architecture tb of system_tb is

    -----------------------------------------------------------------------------
    -- Constants and types
    -----------------------------------------------------------------------------
    constant SEQ_LEN  : integer := 8;
    constant D_MODEL  : integer := 256;
    constant HEAD_DIM : integer := 64;
    constant D_FF     : integer := 512;

    subtype int8_q is signed(7 downto 0);
    type vec_DMODEL is array(0 to D_MODEL - 1) of int8_q;
    type vec_HEAD is array(0 to HEAD_DIM - 1) of int8_q;
    type vec_DFF is array(0 to D_FF - 1) of int8_q;

    -- Clock and reset
    signal clk : std_logic := '0';
    signal rst : std_logic := '1';

    -- Core handshake and data signals
    signal start_mha : std_logic := '0';
    signal done_mha  : std_logic;
    signal mh_out    : vec_DMODEL;

    -- Constants
    constant DATA_WIDTH : integer := 8;
    constant TILE_H     : integer := 12;
    constant TILE_W     : integer := 5;
    constant clk_period : time    := 5 ns;

    -- Declarations at top (inside architecture)
    type mac_result_array is array (0 to 15, 0 to 11) of std_logic_vector(31 downto 0); -- 16×12 = 192
    signal mac_results  : mac_result_array := (others => (others => (others => '0')));
    signal result_index : integer          := 0;

    -- DUT Signals
    signal start : std_logic := '0';
    signal done  : std_logic;

    signal bram_addr     : integer range 0 to 49599     := 0;
    signal bram_data_in  : std_logic_vector(7 downto 0) := (others => '0');
    signal bram_data_out : std_logic_vector(7 downto 0);
    signal bram_wr       : std_logic := '0';

    signal input_a  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0); -- 12x INT32 outputs

    signal start_mac_mha : std_logic;
    signal start_mac_smx : std_logic;
    signal fetch_mac     : std_logic;
    signal done_mac      : std_logic := '0'; -- Simulated handshake

    -- Add Softmax interface signals
    signal start_softmax    : std_logic := '0';
    signal done_softmax     : std_logic;
    signal fetch_qkv        : std_logic := '0';
    signal fetch_attn       : std_logic;
    signal softmax_data_in  : std_logic_vector(12 * 16 - 1 downto 0) := (others => '0');
    signal softmax_data_out : std_logic_vector(12 * DATA_WIDTH - 1 downto 0);

    -- Additional MAC control for softmax
    signal clear_mac_softmax : std_logic;

    -- Debug signals for GTKwave
    signal debug_current_op : string(1 to 4) := "IDLE";

    signal token_bram   : matrix_int8(0 to SEQ_LEN - 1, 0 to D_MODEL - 1);
    signal q_weights    : matrix_int8(0 to HEAD_DIM - 1, 0 to D_MODEL - 1);
    signal k_weights    : matrix_int8(0 to HEAD_DIM - 1, 0 to D_MODEL - 1);
    signal v_weights    : matrix_int8(0 to HEAD_DIM - 1, 0 to D_MODEL - 1);
    signal proj_weights : matrix_int8(0 to D_MODEL - 1, 0 to HEAD_DIM - 1);
    signal fc1_weights  : matrix_int8(0 to D_FF - 1, 0 to D_MODEL - 1);
    signal fc2_weights  : matrix_int8(0 to D_MODEL - 1, 0 to D_FF - 1);

    -- Add MAC arbitration signals
    type mac_arb_state is (IDLE, MHA_ACTIVE, SMX_ACTIVE);
    signal mac_arbiter              : mac_arb_state := IDLE;
    signal start_mac                : std_logic     := '0';
    signal input_a_mha, input_a_smx : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b_mha, input_b_smx : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);

    signal fetch_mac_d1, fetch_mac_d2 : std_logic := '0';
    signal fetch_qkv_enable           : std_logic := '0';
begin

    -- MAC Arbitration FSM with counter for MHA completions
    process (clk, rst)
        variable mha_completion_counter : integer range 0 to 3 := 0;
    begin
        if rst = '1' then
            mac_arbiter <= IDLE;
            mha_completion_counter := 0;
        elsif rising_edge(clk) then
            case mac_arbiter is
                when IDLE =>
                    mha_completion_counter := 0; -- Reset counter when idle
                    if start_mac_mha = '1' then
                        mac_arbiter <= MHA_ACTIVE;
                    elsif start_mac_smx = '1' then
                        mac_arbiter <= SMX_ACTIVE;
                    end if;

                when MHA_ACTIVE =>
                    if done_mha = '1' then
                        if mha_completion_counter = 3 then -- After 4 assertions (0-3)
                            mac_arbiter <= SMX_ACTIVE;
                            mha_completion_counter := 0; -- Reset counter
                        else
                            mha_completion_counter := mha_completion_counter + 1;
                        end if;
                    end if;

                when SMX_ACTIVE =>
                    mha_completion_counter := 0; -- Reset counter when in softmax
                    if done_softmax = '1' then
                        mac_arbiter <= IDLE;
                    end if;
            end case;
        end if;
    end process;

    -- MAC Input Multiplexers
    process (mac_arbiter, input_a_mha, input_b_mha, input_a_smx, input_b_smx)
    begin
        case mac_arbiter is
            when MHA_ACTIVE =>
                input_a   <= input_a_mha;
                input_b   <= input_b_mha;
                start_mac <= start_mac;

            when SMX_ACTIVE =>
                input_a   <= input_a_smx;
                input_b   <= input_b_smx;
                start_mac <= start_mac;

            when IDLE =>
                input_a   <= (others => (others => (others => '0')));
                input_b   <= (others => (others => (others => '0')));
                start_mac <= start_mac_mha or start_mac_smx;
        end case;
    end process;

    -- Instantiate Softmax Core
    softmax_core : entity work.softmax
        generic
        map (
        DATA_WIDTH => DATA_WIDTH
        )
        port map
        (
            clk        => clk,
            rst        => rst,
            start      => start_softmax,
            done       => done_softmax,
            fetch_qkv  => fetch_mac,
            fetch_attn => fetch_attn,
            data_in    => softmax_data_in,
            data_out   => softmax_data_out,

            -- Matrix Mult Interface
            start_mac => start_mac_smx,
            clear_mac => clear_mac_softmax,
            done_mac  => done_mac,
            input_a   => input_a_smx,
            input_b   => input_b_smx,
            output_d  => output_d
        );

    -- Instantiate UUT (multi_head)
    mha_core : entity work.multi_head
        generic
        map (
        DATA_WIDTH => DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map
        (
        clk           => clk,
        rst           => rst,
        start         => start_mha,
        done          => done_mha,
        bram_addr     => bram_addr,
        bram_data_in  => bram_data_in,
        bram_data_out => bram_data_out,
        bram_wr       => bram_wr,
        input_a       => input_a_mha,
        input_b       => input_b_mha,
        start_mac     => start_mac_mha,
        fetch_mac     => fetch_mac,
        done_mac      => done_mac
        );

    -- Instantiate Matrix Multiply Core (MAC Array)
    mac_core : entity work.matrix_mult
        generic
        map (
        DATA_WIDTH => DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_mac_mha or start_mac_smx,
        clear    => fetch_mac or clear_mac_softmax,
        input_a  => input_a,
        input_b  => input_b,
        output_d => output_d,
        done     => done_mac
        );

    -------------------------------------------------------------------------
    -- Clock generation
    -------------------------------------------------------------------------
    clk_process : process
    begin
        loop
            clk <= '0';
            wait for clk_period/2;
            clk <= '1';
            wait for clk_period/2;
        end loop;
    end process;

    -------------------------------------------------------------------------
    -- Stimulus process: load all BRAMs from text files, then drive cores
    -------------------------------------------------------------------------
    stim_proc : process
        -- Stimulus signals
        variable temp_token_bram : matrix_int8(0 to SEQ_LEN - 1, 0 to D_MODEL - 1);
        variable temp_qkv_bram   : matrix_int8(0 to HEAD_DIM - 1, 0 to D_MODEL - 1);
        variable temp_proj_bram  : matrix_int8(0 to D_MODEL - 1, 0 to HEAD_DIM - 1);
        variable temp_fc1_bram   : matrix_int8(0 to D_FF - 1, 0 to D_MODEL - 1);
        variable temp_fc2_bram   : matrix_int8(0 to D_MODEL - 1, 0 to D_FF - 1);
    begin
        -- apply reset
        rst <= '1';
        wait for 20 ns;
        rst              <= '0';
        debug_current_op <= "RST ";

        -- Load entire token embedding BRAM from file
        debug_current_op <= "TOKN";
        read_token_embeddings(
        "/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/token_embeddings.txt", -- Use relative path
        SEQ_LEN, D_MODEL,                                                             -- Correct dimensions
        temp_token_bram
        );

        token_bram <= temp_token_bram;
        wait for clk_period;

        -- Load Q, K, V weight BRAMs
        debug_current_op <= "Q_WT";
        read_int8_matrix(
        "/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/q_weights.txt",
        HEAD_DIM, D_MODEL,
        temp_qkv_bram
        );

        q_weights <= temp_qkv_bram;
        wait for clk_period;

        debug_current_op <= "K_WT";
        read_int8_matrix(
        "/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/k_weights.txt",
        HEAD_DIM, D_MODEL,
        temp_qkv_bram
        );

        k_weights <= temp_qkv_bram;
        wait for clk_period;

        debug_current_op <= "V_WT";
        read_int8_matrix(
        "/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/v_weights.txt",
        HEAD_DIM, D_MODEL,
        temp_qkv_bram
        );

        v_weights <= temp_qkv_bram;

        wait for clk_period;

        -- Write Q weights to BRAM (64x256 matrix, row-major order)
        for row in 0 to HEAD_DIM - 1 loop
            for col in 0 to D_MODEL - 1 loop
                -- Address calculation: row * D_MODEL + col
                bram_addr    <= row * D_MODEL + col;
                bram_data_in <= std_logic_vector(q_weights(row, col));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
        end loop;
        bram_wr <= '0';
        wait for clk_period;

        -- Write K weights to BRAM (64x256 matrix, row-major order)
        for row in 0 to HEAD_DIM - 1 loop
            for col in 0 to D_MODEL - 1 loop
                -- Address calculation: 16384 + (row * D_MODEL + col)
                bram_addr    <= 16384 + (row * D_MODEL + col);
                bram_data_in <= std_logic_vector(k_weights(row, col));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
        end loop;
        bram_wr <= '0';
        wait for clk_period;

        -- Write V weights to BRAM (64x256 matrix, row-major order)
        for row in 0 to HEAD_DIM - 1 loop
            for col in 0 to D_MODEL - 1 loop
                -- Address calculation: 16384 + (row * D_MODEL + col)
                bram_addr    <= 32768 + (row * D_MODEL + col);
                bram_data_in <= std_logic_vector(v_weights(row, col));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
        end loop;
        bram_wr <= '0';
        wait for clk_period;

        -- Write token embeddings to BRAM (addresses 49599-49599-255 downto 0)
        start_softmax <= '1';
        wait for clk_period;
        start_softmax <= '0';

        for token_idx in 0 to 3 loop
            for dim_idx in 0 to D_MODEL - 1 loop
                -- Calculate address (last 256 elements)
                bram_addr    <= (49599 - 255) + dim_idx;
                bram_data_in <= std_logic_vector(token_bram(token_idx, dim_idx));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
            bram_wr <= '0';
            wait for clk_period;

            -- Start the core
            start_mha <= '1';
            wait for clk_period;
            start_mha <= '0';

            -- Transfer output_D to softmax_data_in for this token
            for i in 0 to 17 loop
                wait until fetch_mac = '1';
                wait for clk_period;

                for j in 0 to 11 loop
                    -- Check if value exceeds 16-bit signed range (-32768 to 32767)
                    if signed(output_d((32 * j + 31) downto 32 * j)) > 32767 then
                        softmax_data_in((16 * j + 15) downto 16 * j) <= std_logic_vector(to_signed(32767, 16));

                    elsif signed(output_d((32 * j + 31) downto 32 * j)) <- 32768 then
                        softmax_data_in((16 * j + 15) downto 16 * j) <= std_logic_vector(to_signed(-32767, 16));
                    else
                        -- Value fits in 16 bits: take LSB 16 bits
                        softmax_data_in((16 * j + 15) downto 16 * j) <= output_d((32 * j + 15) downto 32 * j);
                    end if;
                end loop;
                wait for 1 * clk_period;
                softmax_data_in <= (others => '0');
            end loop;

            wait for clk_period;
        end loop;

        debug_current_op <= "SMX ";

        -- Setup softmax input data (example values)
        softmax_data_in <= (others => '0'); -- Replace with actual QKV data
        -- Wait for softmax completion
        wait until done_softmax = '1';

        -- Verify softmax outputs
        debug_current_op <= "CHK ";
        -- Add verification code here
        wait;
    end process;

end architecture;