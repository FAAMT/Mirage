library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.types_pkg.all;

entity ffn_tb is
end ffn_tb;

architecture behavioral of ffn_tb is
    -- Constants
    constant DATA_W     : integer := 17;
    constant CLK_PERIOD : time    := 10 ns;

    -- Component Declaration
    component ffn
        generic
        (
            DATA_W : integer := 17
        );
        port
        (
            clk        : in std_logic;
            rst        : in std_logic;
            start      : in std_logic;
            done       : out std_logic;
            fetch_attn : in std_logic;
            data_in    : in std_logic_vector(16 * DATA_W - 1 downto 0);
            start_mac  : out std_logic;
            clear_mac  : out std_logic;
            done_mac   : in std_logic;
            input_a    : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
            input_b    : out mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
            output_d   : in std_logic_vector(32 * TILE_H - 1 downto 0)
        );
    end component;

    -- Testbench Signals
    signal clk        : std_logic := '0';
    signal rst        : std_logic := '1';
    signal start      : std_logic := '0';
    signal done       : std_logic;
    signal fetch_attn : std_logic                                  := '0';
    signal data_in    : std_logic_vector(16 * DATA_W - 1 downto 0) := (others => '0');
    signal start_mac  : std_logic;
    signal clear_mac  : std_logic;
    signal done_mac   : std_logic := '0';
    signal input_a    : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal input_b    : mac_array(0 to TILE_H - 1, 0 to TILE_W - 1);
    signal output_d   : std_logic_vector(32 * TILE_H - 1 downto 0) := (others => '0');

    -- Hardcoded attention matrix values
    ------------------------------------------------------------------
    -- 85 rows × 3 columns   (255 integers in total)
    ------------------------------------------------------------------
    type attention_matrix is array (0 to 87, 0 to 2) of integer;

    constant ATTENTION_VALUES : attention_matrix :=
    (
    -- Array-0   ─────────────────────────────────────────────
    (126, -1638, 504),    -- row  0
    (-1512, 252, -126),   --      1
    (-630, 756, -2142),   --      2
    (882, -2268, 0),      --      3
    (252, -504, 504),     --      4
    (-1008, 756, -1260),  --      5
    (1008, -1512, 1260),  --      6
    (-1764, 1512, -2016), --      7
    (1764, -2268, 2016),  --      8
    (-2520, 2268, -2772), --      9
    (2520, -3024, 2772),  --     10
    (-3276, 3024, -3528), --     11
    (3276, -3780, 3528),  --     12
    (-4032, 3780, -5040), --     13
    (4032, -4032, 4284),  --     14
    (-5040, 4536, -5418), --     15
    (4788, -5670, 5040),  --     16
    (-6048, 5292, -6300), --     17
    (5544, -6678, 5796),  --     18
    (-6930, 6048, -7308), --     19
    (6300, -7560, 6552),  --     20
    (-7938, 0, 0),        --     21  (pad)

    -- Array-1   ─────────────────────────────────────────────
    (125, -1625, 500),    -- row 22
    (-1500, 250, -125),   -- 23
    (-625, 750, -2125),   -- 24
    (875, -2250, 0),      -- 25
    (250, -500, 500),     -- 26
    (-1000, 750, -1250),  -- 27
    (1000, -1500, 1250),  -- 28
    (-1750, 1500, -2000), -- 29
    (1750, -2250, 2000),  -- 30
    (-2500, 2250, -2750), -- 31
    (2500, -3000, 2750),  -- 32
    (-3250, 3000, -3500), -- 33
    (3250, -3750, 3500),  -- 34
    (-4000, 3750, -5000), -- 35
    (4000, -4000, 4250),  -- 36
    (-5000, 4500, -5375), -- 37
    (4750, -5625, 5000),  -- 38
    (-6000, 5250, -6250), -- 39
    (5500, -6625, 5750),  -- 40
    (-6875, 6000, -7250), -- 41
    (6250, -7500, 6500),  -- 42
    (-7875, 0, 0),        -- 43  (pad)

    -- Array-2   ─────────────────────────────────────────────
    (125, -1625, 500),    -- row 44
    (-1500, 250, -125),   -- 45
    (-625, 750, -2125),   -- 46
    (875, -2250, 0),      -- 47
    (250, -500, 500),     -- 48
    (-1000, 750, -1250),  -- 49
    (1000, -1500, 1250),  -- 50
    (-1750, 1500, -2000), -- 51
    (1750, -2250, 2000),  -- 52
    (-2500, 2250, -2750), -- 53
    (2500, -3000, 2750),  -- 54
    (-3250, 3000, -3500), -- 55
    (3250, -3750, 3500),  -- 56
    (-4000, 3750, -5000), -- 57
    (4000, -4000, 4250),  -- 58
    (-5000, 4500, -5375), -- 59
    (4750, -5625, 5000),  -- 60
    (-6000, 5250, -6250), -- 61
    (5500, -6625, 5750),  -- 62
    (-6875, 6000, -7250), -- 63
    (6250, -7500, 6500),  -- 64
    (-7875, 0, 0),        -- 65  (pad)

    -- Array-3   ─────────────────────────────────────────────
    (125, -1625, 500),    -- row 66
    (-1500, 250, -125),   -- 67
    (-625, 750, -2125),   -- 68
    (875, -2250, 0),      -- 69
    (250, -500, 500),     -- 70
    (-1000, 750, -1250),  -- 71
    (1000, -1500, 1250),  -- 72
    (-1750, 1500, -2000), -- 73
    (1750, -2250, 2000),  -- 74
    (-2500, 2250, -2750), -- 75
    (2500, -3000, 2750),  -- 76
    (-3250, 3000, -3500), -- 77
    (3250, -3750, 3500),  -- 78
    (-4000, 3750, -5000), -- 79
    (4000, -4000, 4250),  -- 80
    (-5000, 4500, -5375), -- 81
    (4750, -5625, 5000),  -- 82
    (-6000, 5250, -6250), -- 83
    (5500, -6625, 5750),  -- 84
    (-6875, 6000, -7250),
    (6250, -7500, 6500),
    (-7875, 0, 0) -- 85  
    );

begin
    -- Clock generation
    clk <= not clk after CLK_PERIOD / 2;

    -- Instantiate DUT
    dut : ffn
    generic
    map (
    DATA_W => DATA_W
    )
    port map
    (
        clk        => clk,
        rst        => rst,
        start      => start,
        done       => done,
        fetch_attn => fetch_attn,
        data_in    => data_in,
        start_mac  => start_mac,
        clear_mac  => clear_mac,
        done_mac   => done_mac,
        input_a    => input_a,
        input_b    => input_b,
        output_d   => output_d
    );

    -- Stimulus process
    stim_proc : process
        variable a_idx, e_idx : integer := 0;
        variable data         : signed(31 downto 0);
    begin
        -- Reset
        rst <= '1';
        wait for 100 ns;
        rst <= '0';
        wait for CLK_PERIOD;

        ------------------------------------------------------------------
        -- FIRST 12 words  (pass 0 : ATTENTION_VALUES rows 0-2)
        ------------------------------------------------------------------
        for pass in 0 to 21 loop
            report "Loading 3 values" & " Pass: " & integer'image(pass);

            fetch_attn <= '1';

            if pass < 21 then
                for row in 0 to 11 loop -- 12×32-bit words
                    -- map 3-row groups to the four output arrays
                    a_idx := pass + (22 * (row / 3)); -- 0-3  (array index)
                    e_idx := (row mod 3);             -- 0-2  (element in that array)

                    -- ATTENTION_VALUES rows 0-2   (use e_idx)
                    data := to_signed(ATTENTION_VALUES(a_idx, e_idx), 32);
                    output_d(32 * row + 31 downto 32 * row) <= std_logic_vector(data);

                    report " Pass: " & integer'image(pass) &
                        " Row: " & integer'image(row) &
                        " Array: " & integer'image(a_idx) &
                        " Element: " & integer'image(e_idx) &
                        " Val:  " & integer'image(ATTENTION_VALUES(a_idx, e_idx));
                end loop;
            else
                for row in 0 to 11 loop -- 12×32-bit words
                    if (row = 0) or (row = 3) or (row = 6) or (row = 9) then
                        -- map 3-row groups to the four output arrays
                        a_idx := pass + 22 * (row / 3); -- 0-3  (array index)
                        e_idx := (row mod 3);           -- 0-2  (element in that array)

                        -- ATTENTION_VALUES rows 0-2   (use e_idx)
                        data := to_signed(ATTENTION_VALUES(a_idx, e_idx), 32);
                        output_d(32 * row + 31 downto 32 * row) <= std_logic_vector(data);

                        report " Pass: " & integer'image(pass) &
                            " Row: " & integer'image(row) &
                            " Array: " & integer'image(a_idx) &
                            " Element: " & integer'image(e_idx) &
                            " Val:  " & integer'image(ATTENTION_VALUES(a_idx, e_idx));
                    else
                        output_d(32 * row + 31 downto 32 * row) <= (others => '0');
                    end if;
                end loop;
            end if;

            wait for CLK_PERIOD;
            fetch_attn <= '0';
            wait for 5 * CLK_PERIOD;

        end loop;

        wait;
    end process;
end architecture;