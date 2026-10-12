class CostModel:
    """
    Applies spread and slippage to trade entries and exits.
    Makes every price worse, exactly as real life does.
    """
    def __init__(self, config: dict):
        self.spread = config.get('spread', {})
        self.slippage = config.get('slippage', {})

    def _get_pip_value(self, pair: str) -> float:
        """Returns the price value of 1 pip for a given pair."""
        if pair == "USDJPY":
            return 0.01
        return 0.0001

    def get_adjusted_entry(self, pair: str, raw_price: float, direction: str) -> float:
        """
        Applies half the round-trip spread to the entry price.
        Longs buy higher, Shorts sell lower.
        """
        spread_pips = self.spread.get(pair, 1.5)
        half_spread_pips = spread_pips / 2.0
        pip_val = self._get_pip_value(pair)
        cost = half_spread_pips * pip_val
        
        if direction == "long":
            return raw_price + cost 
        else:
            return raw_price - cost 

    def get_adjusted_exit(self, pair: str, raw_price: float, direction: str, exit_type: str) -> float:
        """
        Applies half the spread + slippage to the exit price.
        Longs sell lower, Shorts buy higher.
        
        exit_type must be: 'take_profit', 'stop_loss', or 'time_stop'
        """
        # 1. Calculate Spread Cost
        spread_pips = self.spread.get(pair, 1.5)
        half_spread_pips = spread_pips / 2.0
        pip_val = self._get_pip_value(pair)
        spread_cost = half_spread_pips * pip_val
        
        # 2. Calculate Slippage Cost
        slip_pips = self.slippage.get(exit_type, 1.0) # Default to 1 pip if type unknown
        slip_cost = slip_pips * pip_val
        
        total_cost = spread_cost + slip_cost
        
        if direction == "long":
            return raw_price - total_cost 
        else:
            return raw_price + total_cost 