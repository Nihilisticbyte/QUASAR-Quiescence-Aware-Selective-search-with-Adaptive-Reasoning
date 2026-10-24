# Test position data

- `wac.epd` — Win At Chess suite, 300 tactical positions (`bm` = book best
  move(s), `id` = WAC.001 ... WAC.300). Copied from the `wacnew.epd` file in
  https://github.com/fsmosca/EPD-Analyzer (epd folder), with Windows line
  endings converted to plain newlines. The positions are from the book
  _Winning Chess Positions_ (Reinfeld); the repo itself is GPL-3.0, so credit
  it in the report's references.
- Real-game positions are NOT stored here: put a PGN of real games
  (e.g. `data/real_games.pgn`) and sample from it with
  `engine.positions.sample_pgn_positions`.
