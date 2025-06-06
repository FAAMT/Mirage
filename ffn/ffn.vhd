library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity ffn is
    generic
    (
        DATA_WIDTH : integer := 17
    );
    port
    (
        clk : in std_logic;
        rst : in std_logic;

        start : in std_logic;
        done  : out std_logic;

        fetch_attn : in std_logic;
        data_in    : in std_logic_vector(16 * DATA_WIDTH - 1 downto 0);

        -- BRAM Signals
        bram_addr     : in integer; -- 49151 + 191 + 255
        bram_data_in  : in std_logic_vector(7 downto 0);
        bram_data_out : out std_logic_vector(7 downto 0);
        bram_wr       : in std_logic;

        -- Matrix Mult Control Interface
        start_mac : out std_logic;
        clear_mac : out std_logic;
        done_mac  : in std_logic;

        -- Matrix Mult Data Interface
        input_a  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        input_b  : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
        output_d : in std_logic_vector(32 * TILE_H - 1 downto 0)
    );
end entity;

architecture rtl of ffn is
    -------------------------------------------------------------------
    -- 1) TYPE DECLARATIONS
    -------------------------------------------------------------------
    type state_type is (
        idle, compute_pl, compute_hl, compute_ol, computer_output_64, result
    );

    type bram_attention_type is array(0 to 255) of signed(17 downto 0);
    type bram_hl_in_type is array(0 to 256 * 4 - 1) of signed(15 downto 0); -- 4 outputs, 256 element vec
    type bram_ol_in_type is array(0 to 512 * 1 - 1) of signed(15 downto 0); -- 4 outputs, 256 element vec

    type bram_proj_type is array(0 to 256 * 64 - 1) of signed(7 downto 0);
    type bram_hlw_type is array(0 to 256 * 512 - 1) of signed(7 downto 0); -- 512 neurons, 256 elemevt vec
    type bram_olw_type is array(0 to 512 * 256 - 1) of signed(7 downto 0); -- 256 neurons, 256 elemevt vec

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    signal bram_attention : bram_attention_type;
    signal bram_proj      : bram_proj_type;
    signal bram_hl_in     : bram_hl_in_type;
    signal bram_ol_in     : bram_ol_in_type;
    signal bram_hlw       : bram_hlw_type;
    signal bram_olw       : bram_olw_type;

    signal att_fetch_ctr : integer := 0;
    signal hl_fetch_ctr  : integer := 0;
    signal ol_fetch_ctr  : integer := 0;

    signal proj_ext_iter   : integer := 0;
    signal proj_int_iter   : integer := 0;
    signal hidden_ext_iter : integer := 0;
    signal hidden_int_iter : integer := 0;
    signal outer_ext_iter  : integer := 0;
    signal outer_int_iter  : integer := 0;

begin

    -- Projection Layer BRAM Write Process
    process (clk, rst)
    begin
        if rst = '1' then
            bram_proj <= (others => (others => '0'));
            bram_hlw  <= (others => (others => '0'));
        elsif rising_edge(clk) then
            if bram_wr = '1' then
                if bram_addr >= 0 and bram_addr < 16384 then
                    bram_proj(bram_addr) <= signed(bram_data_in);
                elsif bram_addr < 147456 then
                    bram_hlw(bram_addr - 16384) <= signed(bram_data_in);
                elsif bram_addr < 278528 then
                    bram_olw(bram_addr - 147456) <= signed(bram_data_in);
                end if;
            end if;
        end if;
    end process;

    -- Attention BRAM Fetch Process
    process (clk, rst)
        variable r_mod_3 : integer;
        variable r_div_3 : integer;
    begin
        if rst = '1' then
            -- BRAM Signals
            bram_attention <= (others => (others => '0'));

            -- Control Signals
            att_fetch_ctr <= 0;

        elsif rising_edge(clk) then
            if fetch_attn = '1' then
                att_fetch_ctr <= att_fetch_ctr + 1;

                if att_fetch_ctr < 21 then
                    for row in 0 to 11 loop
                        r_mod_3 := row mod 3;
                        r_div_3 := row / 3;

                        bram_attention(r_div_3 * 64 + r_mod_3 + 3 * att_fetch_ctr) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 18);
                    end loop;
                else
                    for row in 0 to 11 loop
                        if (row = 0) or (row = 3) or (row = 6) or (row = 9) then
                            r_mod_3 := row mod 3;
                            r_div_3 := row / 3;

                            bram_attention(r_div_3 * 64 + r_mod_3 + 3 * att_fetch_ctr) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 18);
                        end if;
                    end loop;
                end if;
            end if;
        end if;
    end process;

    -- Hidden Layer BRAM Fetch Process
    process (clk, rst)
        variable r_mod_3 : integer;
        variable r_div_3 : integer;
    begin
        if rst = '1' then
            -- BRAM Signals
            bram_hl_in <= (others => (others => '0'));

        elsif rising_edge(clk) then
            for row in 0 to 11 loop
                r_mod_3 := row mod 3;
                r_div_3 := row / 3;

                if state = compute_pl and clear_mac = '1' then
                    bram_hl_in(256 * r_div_3 + r_mod_3 + 3 * (proj_ext_iter - 1)) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 16);
                elsif state = compute_hl and prev_state = compute_pl and proj_ext_iter = 85 and proj_int_iter = 12 then
                    if (row = 0 or row = 3 or row = 6 or row = 9) then
                        bram_hl_in(256 * r_div_3 + r_mod_3 + 3 * 85) <= resize(signed(output_d(32 * row + 31 downto 32 * row)), 16);
                    end if;
                end if;
            end loop;
        end if;
    end process;

    -- Hidden Layer BRAM Fetch Process with ReLU
    process (clk, rst)
    begin
        if rst = '1' then
            -- BRAM Signals
            bram_ol_in <= (others => (others => '0'));
        elsif rising_edge(clk) then
            for row in 0 to 11 loop
                if hidden_ext_iter = 42 then
                    if state = compute_ol and prev_state = compute_hl and clear_mac = '1' then
                        if row < 8 then
                            -- Apply ReLU: if negative, set to zero
                            if signed(output_d(32 * row + 31 downto 32 * row)) < 0 then
                                bram_ol_in(row + (12 * hidden_ext_iter)) <= (others => '0');
                            else
                                bram_ol_in(row + (12 * hidden_ext_iter)) <=
                                resize(signed(output_d(32 * row + 31 downto 32 * row)), 16);
                            end if;
                        end if;
                    end if;
                else
                    if state = compute_hl and prev_state = compute_hl and clear_mac = '1' then
                        -- Apply ReLU: if negative, set to zero
                        if signed(output_d(32 * row + 31 downto 32 * row)) < 0 then
                            bram_ol_in(row + (12 * hidden_ext_iter)) <= (others => '0');
                        else
                            bram_ol_in(row + (12 * hidden_ext_iter)) <=
                            resize(signed(output_d(32 * row + 31 downto 32 * row)), 16);
                        end if;
                    end if;
                end if;
            end loop;
        end if;
    end process;

    -- Matrix Mult Tiling
    process (clk, rst)
        variable r_mod_3 : integer;
        variable r_div_3 : integer;
    begin
        if rst = '1' then
            state      <= idle;
            prev_state <= idle;
            start_mac  <= '0';
            clear_mac  <= '0';

            proj_int_iter   <= 0;
            proj_ext_iter   <= 0;
            hidden_int_iter <= 0;
            hidden_ext_iter <= 0;
            outer_int_iter  <= 0;
            outer_ext_iter  <= 0;

            for row in 0 to 11 loop
                for col in 0 to 4 loop
                    input_a(row, col) <= (others => '0');
                    input_b(row, col) <= (others => '0');
                end loop;
            end loop;

        elsif rising_edge(clk) then
            case state is
                when idle =>
                    prev_state <= idle;
                    clear_mac  <= '0';

                    if start = '1' then
                        state <= compute_pl;
                    end if;

                when compute_pl =>
                    prev_state <= compute_pl;

                    if start_mac = '0' and done_mac = '1' then
                        clear_mac <= '0';

                        if proj_int_iter = 12 and done_mac = '1'then
                            if proj_ext_iter = 85 then
                                state     <= compute_hl;
                                clear_mac <= '1';
                            else
                                proj_ext_iter <= proj_ext_iter + 1;
                                proj_int_iter <= 0;
                                clear_mac     <= '1';
                            end if;
                        else
                            proj_int_iter <= proj_int_iter + 1;
                        end if;

                    elsif start_mac = '0' and done_mac = '0' then
                        start_mac <= '1';
                        clear_mac <= '0';
                    else
                        start_mac <= '0';
                        clear_mac <= '0';
                    end if;

                    for row in 0 to 11 loop
                        for col in 0 to 4 loop
                            r_mod_3 := row mod 3;
                            r_div_3 := row / 3;

                            if proj_ext_iter = 85 then
                                if (row = 0 or row = 3 or row = 6 or row = 9) then
                                    if proj_int_iter < 12 then
                                        input_a(row, col) <= resize(shift_right(bram_attention(64 * r_div_3 + col + 5 * proj_int_iter), 8), 8);
                                        input_b(row, col) <= bram_proj(r_mod_3 + 256 * col + 256 * 5 * proj_int_iter + 3 * proj_ext_iter);
                                    elsif col < 4 then
                                        input_a(row, col) <= resize(shift_right(bram_attention(64 * r_div_3 + col + 5 * proj_int_iter), 8), 8);
                                        input_b(row, col) <= bram_proj(r_mod_3 + 256 * col + 256 * 5 * proj_int_iter + 3 * proj_ext_iter);
                                    else
                                        input_a(row, col) <= (others => '0');
                                        input_b(row, col) <= (others => '0');
                                    end if;
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');
                                end if;
                            else
                                if proj_int_iter < 12 then
                                    input_a(row, col) <= resize(shift_right(bram_attention(64 * r_div_3 + col + 5 * proj_int_iter), 8), 8);
                                    input_b(row, col) <= bram_proj(r_mod_3 + 256 * col + 256 * 5 * proj_int_iter + 3 * proj_ext_iter);
                                elsif col < 4 then
                                    input_a(row, col) <= resize(shift_right(bram_attention(64 * r_div_3 + col + 5 * proj_int_iter), 8), 8);
                                    input_b(row, col) <= bram_proj(r_mod_3 + 256 * col + 256 * 5 * proj_int_iter + 3 * proj_ext_iter);
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');
                                end if;
                            end if;
                        end loop;
                    end loop;

                when compute_hl =>
                    prev_state <= compute_hl;

                    if start_mac = '0' and done_mac = '1' then
                        clear_mac <= '0';

                        if hidden_int_iter = 51 and done_mac = '1'then
                            if hidden_ext_iter = 42 then
                                state     <= compute_ol;
                                clear_mac <= '1';
                            else
                                hidden_int_iter <= 0;
                                clear_mac       <= '1';
                            end if;
                        else
                            hidden_int_iter <= hidden_int_iter + 1;
                        end if;

                    elsif start_mac = '0' and done_mac = '0' then
                        start_mac <= '1';
                        clear_mac <= '0';

                        if hidden_int_iter = 0 and prev_state = compute_hl and state = compute_hl then
                            hidden_ext_iter <= hidden_ext_iter + 1;
                        end if;
                    else
                        start_mac <= '0';
                        clear_mac <= '0';
                    end if;

                    for row in 0 to 11 loop
                        for col in 0 to 4 loop
                            r_mod_3 := row mod 3;
                            r_div_3 := row / 3;

                            if hidden_ext_iter = 42 then
                                if row < 8 then
                                    if hidden_int_iter < 51 then
                                        input_a(row, col) <= resize(shift_right(signed(bram_hl_in(col + 5 * hidden_int_iter)), 8), 8);
                                        input_b(row, col) <= bram_hlw(col + 5 * hidden_int_iter + 256 * row + 12 * 256 * hidden_ext_iter);
                                    else
                                        input_a(row, col) <= (others => '0');
                                        input_b(row, col) <= (others => '0');

                                        if col < 1 then
                                            input_a(row, col) <= resize(shift_right(signed(bram_hl_in(col + 5 * hidden_int_iter)), 8), 8);
                                            input_b(row, col) <= bram_hlw(col + 5 * hidden_int_iter + 256 * row + 12 * 256 * hidden_ext_iter);
                                        end if;
                                    end if;
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');
                                end if;
                            else
                                if hidden_int_iter < 51 then
                                    input_a(row, col) <= resize(shift_right(signed(bram_hl_in(col + 5 * hidden_int_iter)), 8), 8);
                                    input_b(row, col) <= bram_hlw(col + 5 * hidden_int_iter + 256 * row + 12 * 256 * hidden_ext_iter);
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');

                                    if col < 1 then
                                        input_a(row, col) <= resize(shift_right(signed(bram_hl_in(col + 5 * hidden_int_iter)), 8), 8);
                                        input_b(row, col) <= bram_hlw(col + 5 * hidden_int_iter + 256 * row + 12 * 256 * hidden_ext_iter);
                                    end if;
                                end if;
                            end if;
                        end loop;
                    end loop;

                when compute_ol =>
                    prev_state <= compute_ol;

                    if start_mac = '0' and done_mac = '1' then
                        clear_mac <= '0';

                        if outer_int_iter = 102 and done_mac = '1'then
                            if outer_ext_iter = 21 then
                                state     <= idle;
                                clear_mac <= '1';
                            else
                                outer_int_iter <= 0;
                                clear_mac      <= '1';
                            end if;
                        else
                            outer_int_iter <= outer_int_iter + 1;
                        end if;

                    elsif start_mac = '0' and done_mac = '0' then
                        start_mac <= '1';
                        clear_mac <= '0';

                        if outer_int_iter = 0 and prev_state = compute_ol and state = compute_ol then
                            outer_ext_iter <= outer_ext_iter + 1;
                        end if;
                    else
                        start_mac <= '0';
                        clear_mac <= '0';
                    end if;

                    for row in 0 to 11 loop
                        for col in 0 to 4 loop
                            r_mod_3 := row mod 3;
                            r_div_3 := row / 3;

                            if outer_ext_iter = 21 then
                                if row < 4 then
                                    if outer_int_iter < 102 then
                                        input_a(row, col) <= resize(shift_right(bram_ol_in(col + 5 * outer_int_iter), 4), 8);
                                        input_b(row, col) <= bram_olw(col + 5 * outer_int_iter + 512 * row + 12 * 512 * outer_ext_iter);
                                    else
                                        input_a(row, col) <= (others => '0');
                                        input_b(row, col) <= (others => '0');

                                        if col < 2 then
                                            input_a(row, col) <= resize(shift_right(bram_ol_in(col + 5 * outer_int_iter), 4), 8);
                                            input_b(row, col) <= bram_olw(col + 5 * outer_int_iter + 512 * row + 12 * 512 * outer_ext_iter);
                                        end if;
                                    end if;
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');
                                end if;
                            else
                                if outer_int_iter < 102 then
                                    input_a(row, col) <= resize(shift_right(bram_ol_in(col + 5 * outer_int_iter), 4), 8);
                                    input_b(row, col) <= bram_olw(col + 5 * outer_int_iter + 512 * row + 12 * 512 * outer_ext_iter);
                                else
                                    input_a(row, col) <= (others => '0');
                                    input_b(row, col) <= (others => '0');

                                    if col < 2 then
                                        input_a(row, col) <= resize(shift_right(bram_ol_in(col + 5 * outer_int_iter), 4), 8);
                                        input_b(row, col) <= bram_olw(col + 5 * outer_int_iter + 512 * row + 12 * 512 * outer_ext_iter);
                                    end if;
                                end if;
                            end if;
                        end loop;
                    end loop;
                when others =>
                    state <= idle;
            end case;

        end if;
    end process;

end architecture;