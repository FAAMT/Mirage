library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use STD.TEXTIO.all;
use IEEE.STD_LOGIC_TEXTIO.all;
use work.types_pkg.all;
use work.tb_io_pkg.all;

entity multi_head_tb is
end entity;

architecture behavior of multi_head_tb is
    -- Declarations at top (inside architecture)
    type mac_result_array is array (0 to 15, 0 to 11) of std_logic_vector(31 downto 0); -- 16×12 = 192
    signal mac_results : mac_result_array := (others => (others => (others => '0')));

    type matrix_int8_t is array (0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1) of signed(7 downto 0);
    signal result_index : integer := 0;

    -- DUT Signals
    signal clk   : std_logic := '0';
    signal rst   : std_logic := '0';
    signal start : std_logic := '0';
    signal done  : std_logic;

    signal bram_addr     : integer range 0 to (3 * (HEAD_DIM * MODEL_DIM) + TOKEN_DIM) - 1 := 0;
    signal bram_data_in  : std_logic_vector(7 downto 0)                                    := (others => '0');
    signal bram_data_out : std_logic_vector(7 downto 0);
    signal bram_wr       : std_logic := '0';

    signal input_a  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b  : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d : std_logic_vector(32 * TILE_H - 1 downto 0); -- 12x INT32 outputs

    signal start_mac : std_logic;
    signal fetch_mac : std_logic;
    signal done_mac  : std_logic := '0'; -- Simulated handshake

begin

    -- Instantiate UUT (multi_head)
    uut : entity work.multi_head
        generic
        map (
        DATA_WIDTH => MHA_DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port map
        (
            clk           => clk,
            rst           => rst,
            start         => start,
            done          => done,
            bram_addr     => bram_addr,
            bram_data_in  => bram_data_in,
            bram_data_out => bram_data_out,
            bram_wr       => bram_wr,
            input_a       => input_a,
            input_b       => input_b,
            start_mac     => start_mac,
            fetch_mac     => fetch_mac,
            done_mac      => done_mac
        );

    -- Instantiate Matrix Multiply Core (MAC Array)
    mac_core : entity work.matrix_mult
        generic
        map (
        DATA_WIDTH => MM_IN_DATA_WIDTH,
        TILE_H     => TILE_H,
        TILE_W     => TILE_W
        )
        port
        map (
        clk      => clk,
        rst      => rst,
        start    => start_mac,
        clear    => fetch_mac,
        input_a  => input_a,
        input_b  => input_b,
        output_d => output_d, -- output not needed yet for this phase
        done     => done_mac
        );

    -- Clock generation
    clk_process : process
    begin
        loop
            clk <= '0';
            wait for clk_period / 2;
            clk <= '1';
            wait for clk_period / 2;
        end loop;
    end process;

    -- Stimulus process
    stim_proc : process
        variable rand_val    : integer;
        variable line_buffer : line;
        file output_file     : text open write_mode is "q_output.txt";

        ---------------------------------------------------------
        -- ▶ Load token embeddings & QKV weights from text files --
        ---------------------------------------------------------
        variable token_mat : matrix_int8(0 to 0, 0 to TOKEN_DIM - 1);
        variable q_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable k_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
        variable v_mat     : matrix_int8(0 to HEAD_DIM - 1, 0 to MODEL_DIM - 1);
    begin
        -- Reset
        rst <= '1';
        wait for 20 ns;
        rst <= '0';

        for tok_num in 0 to 5 loop
            -- Process all 8 heads in a loop
            read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/token_embeddings_" & integer'image(tok_num) & ".txt",
            1, TOKEN_DIM, token_mat);

            -------------------------------------------------------------------
            -- ▶ 1. Token Embeddings 
            -------------------------------------------------------------------
            for c in 0 to TOKEN_DIM - 1 loop
                -- Calculate address (last 256 elements)
                bram_addr    <= (3 * HEAD_DIM * MODEL_DIM) + c;
                bram_data_in <= std_logic_vector(token_mat(0, c));
                bram_wr      <= '1';
                wait for clk_period;
            end loop;
            bram_wr <= '0';
            wait for clk_period;
            wait for 2 * clk_period;

            for head_num in 0 to 1 loop
                -- Read weight matrices for current head
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_query_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, q_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_key_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, k_mat);
                read_int8_matrix("/Users/faamt/Desktop/Nomadic Inc/Mirage/model/tb_stim/layer0_value_head" & integer'image(head_num) & ".txt",
                HEAD_DIM, MODEL_DIM, v_mat);

                -------------------------------------------------------------------
                -- ▶ 2. Q weights (layer-0, head-N) 0 … 16 383
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        bram_addr    <= Q_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(q_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
                wait for 2 * clk_period;

                -------------------------------------------------------------------
                -- ▶ 3. K weights 16 384 … 32 767
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        bram_addr    <= K_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(k_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
                wait for 2 * clk_period;

                -------------------------------------------------------------------
                -- ▶ 4. V weights 32 768 … 49 151
                -------------------------------------------------------------------
                for r in 0 to HEAD_DIM - 1 loop
                    for c in 0 to WEIGHT_SLICE - 1 loop
                        bram_addr    <= V_BASE + r * WEIGHT_SLICE + c;
                        bram_data_in <= std_logic_vector(v_mat(r, c));
                        bram_wr      <= '1';
                        wait for clk_period;
                    end loop;
                end loop;
                bram_wr <= '0';
                wait for 2 * clk_period;

                -- Start the core
                start <= '1';
                wait for clk_period;
                start <= '0';

                wait until done = '1';
                wait for 2 * clk_period;
            end loop;
        end loop;

        -- -- Capture 6 fetch_mac outputs
        -- for pass in 0 to 17 loop -- 18 passes × 12 = 192
        --     wait until fetch_mac = '1';
        --     wait for clk_period;

        --     for i in 0 to 11 loop
        --         mac_results(result_index mod 6, i) <= output_d((i + 1) * 32 - 1 downto i * 32);
        --     end loop;
        --     result_index <= result_index + 1;

        --     wait for clk_period;

        --     for i in 0 to 11 loop
        --         if pass * 12 + i < 1216 then
        --             line_buffer := null;
        --             write(line_buffer, integer'image(to_integer(signed(mac_results(pass mod 6, i)))));
        --             writeline(output_file, line_buffer);
        --         end if;
        --     end loop;

        -- end loop;

        wait for 10 * clk_period; -- small stabilization
        wait;
    end process;

end architecture;