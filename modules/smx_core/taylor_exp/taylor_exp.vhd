library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity taylor_exp is
    generic
    (
        DATA_W : integer := SCORE_BRAM_DATA_W
    );
    port
    (
        -- Clk & Rst 
        clk : in std_logic;
        rst : in std_logic;

        -- Control Signals
        start      : in std_logic;
        done       : out std_logic;
        fetch_attn : out std_logic;

        -- BRAM Interface
        bram_rd    : out std_logic;
        addr_off_0 : out integer;

        -- IO Interface
        data_in  : in score_io_type;
        data_out : out std_logic_vector(TILE_H * DATA_W - 1 downto 0)
    );
end entity;

architecture rtl of taylor_exp is

    -------------------------------------------------------------------
    -- 1) TYPE DECLARATIONS
    -------------------------------------------------------------------
    type state_type is (
        idle, compute_max, compute_taylor
    );

    -- Max Finder Types
    type max_val_type is array(0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(DATA_W - 1 downto 0);
    type max_curr_type is array(0 to NUM_HEADS - 1) of signed(DATA_W - 1 downto 0);

    -- Taylor Expansion Types
    type taylor_array_21 is array(0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(DATA_W - 1 downto 0);
    type taylor_array_42 is array(0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(2 * DATA_W - 1 downto 0);
    type taylor_array_63 is array(0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(3 * DATA_W - 1 downto 0);
    type taylor_array_approx is array(0 to TOK_CONTEXT * NUM_HEADS - 1) of signed(63 downto 0);

    -------------------------------------------------------------------
    -- 2) SIGNAL DECLARATIONS (exactly as you provided)
    -------------------------------------------------------------------
    signal state      : state_type := idle;
    signal prev_state : state_type := idle;

    -- BRAM Max Value
    signal bram_max : bram_scores_type;

    -- Start/Done Signals
    signal start_max    : std_logic;
    signal start_taylor : std_logic;
    signal done_max     : std_logic;
    signal done_taylor  : std_logic;

    -- Max Finder Signals
    signal max_val     : max_val_type;
    signal current_max : max_curr_type;
    signal max_index   : integer := 0;
    signal tok_index   : integer := 0;
    signal max_ctr     : integer := 0;
    signal delay_ctr   : integer := 0;
    signal index       : integer range 0 to 63;
    signal processing  : std_logic;
    signal find_max    : std_logic;

    -- Taylor Expansion Signals
    signal z_diff        : taylor_array_21     := (others => (others => '0'));
    signal z_diff_sq     : taylor_array_42     := (others => (others => '0'));
    signal z_diff_cu     : taylor_array_63     := (others => (others => '0'));
    signal term1         : taylor_array_approx := (others => (others => '0'));
    signal term2         : taylor_array_approx := (others => (others => '0'));
    signal term3         : taylor_array_approx := (others => (others => '0'));
    signal taylor_approx : taylor_array_approx := (others => (others => '0'));
    signal taylor_sum    : signed(63 downto 0);

    signal taylor_exp_ready : std_logic;
    signal taylor_exp_ctr   : integer;
    -- signal taylor_exp_out   : taylor_exp_array_out;

    signal store_attn : std_logic;
begin
    process (clk, rst)
    begin
        if rst = '1' then
            state <= idle;

        elsif rising_edge(clk) then
            case state is
                when idle =>
                    -- Reset Max Finder & Taylor Expansion
                    start_max    <= '0';
                    start_taylor <= '0';
                    store_attn   <= '0';

                    -- Reset Done Signals
                    done <= '0';

                    if start = '1' then
                        state <= compute_max;
                    end if;

                when compute_max =>
                    if done_max = '1' then
                        state <= compute_taylor;
                    end if;

                when compute_taylor =>
                    if done_taylor = '1' then
                        state <= idle;
                    end if;

                when others =>
                    state <= idle;
            end case;
        end if;
    end process;

    -------------------------------------------------------------------
    -- MAX FINDER
    -------------------------------------------------------------------
    process (start, max_index, data_in)
    begin
        for i in 0 to NUM_HEADS - 1 loop
            current_max(i) <= data_in(TOK_CONTEXT * i + max_index);
        end loop;
    end process;

    process (clk, rst)
        variable temp_max  : max_curr_type;
        variable temp_addr : integer := 0;
    begin
        if rst = '1' then
            -- Max Finder Signals
            max_val   <= (others => (others => '0'));
            max_index <= 0;
            done_max  <= '0';

        elsif rising_edge(clk) then
            if state = compute_max then
                done_max <= '0';

                -- Compare Value Logic
                for i in 0 to NUM_HEADS - 1 loop
                    temp_max(i) := shift_right(current_max(i), 3);
                    temp_addr   := NUM_HEADS * tok_index + i;

                    if max_val(temp_addr) < temp_max(i) then
                        max_val(temp_addr) <= temp_max(i);
                    end if;
                end loop;

                if max_index = TOK_CONTEXT - 1 then
                    -- Reset Index
                    max_index <= 0;
                    done_max  <= '1';
                else
                    if max_index = 0 then
                        -- Set Initial Value
                        for i in 0 to NUM_HEADS - 1 loop
                            max_val(temp_addr) <= temp_max(i);
                        end loop;
                    end if;

                    -- Increment Index
                    max_index <= max_index + 1;
                end if;
            end if;
        end if;
    end process;

    -------------------------------------------------------------------
    -- TAYLOR APPROX
    -------------------------------------------------------------------
    process (clk, rst)
        variable temp_data : score_io_type;
        variable temp_addr : integer := 0;
    begin
        if rst = '1' then
            z_diff           <= (others => (others => '0'));
            z_diff_sq        <= (others => (others => '0'));
            z_diff_cu        <= (others => (others => '0'));
            term1            <= (others => (others => '0'));
            term2            <= (others => (others => '0'));
            term3            <= (others => (others => '0'));
            taylor_sum       <= (others => '0');
            taylor_exp_ready <= '0';
        elsif rising_edge(clk) then
            if state = compute_taylor then
                for i in 0 to NUM_HEADS - 1 loop
                    for j in 0 to TOK_CONTEXT - 1 loop
                        -- Temporary Values
                        temp_addr            := TOK_CONTEXT * i + j;
                        temp_data(temp_addr) := shift_right(data_in(temp_addr), 3);

                        -- (Z'), (Z')^2, (Z')^3 Values
                        z_diff(temp_addr)    <= temp_data(temp_addr) - max_val(i);
                        z_diff_sq(temp_addr) <= z_diff(temp_addr) * z_diff(temp_addr);
                        z_diff_cu(temp_addr) <= z_diff_sq(temp_addr) * z_diff(temp_addr);

                        -- Intermediary Terms
                        term1(temp_addr) <= resize(z_diff(temp_addr), 64);
                        term2(temp_addr) <= shift_right(resize(z_diff_sq(temp_addr), 64), 1);
                        term3(temp_addr) <= shift_right(resize(z_diff_cu(temp_addr) * to_signed(174763, 64), 64), 20);

                        taylor_approx(temp_addr) <= term1(i) + term2(i) + term3(i);
                    end loop;
                end loop;
            end if;

            -- if taylor_done = '1' then
            --     taylor_exp_ready <= '1';
            --     taylor_sum       <= resize(taylor_approx(0), 64)
            --         + resize(taylor_approx(1), 64)
            --         + resize(taylor_approx(2), 64)
            --         + resize(taylor_approx(3), 64);
            -- else
            --     taylor_exp_ready <= '0';
            -- end if;
        end if;
    end process;

    -- process (taylor_approx, taylor_sum)
    -- begin
    --     for i in 0 to 3 loop
    --         if taylor_sum /= 0 then
    --             taylor_exp_out(i) <= shift_left(to_signed(1, 64), 7) - abs((shift_left(taylor_approx(i), 7)) / taylor_sum) - 1;

    --         else
    --             taylor_exp_out(i) <= (others => '0');
    --         end if;
    --     end loop;
    -- end process;

    -------------------------------------------------------------------
    -- ATTENTION WRITEBACK
    -------------------------------------------------------------------
    fetch_attn <= store_attn; -- Trigger for FFN 

end architecture;