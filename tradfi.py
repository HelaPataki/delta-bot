"""Фьючерсы Binance на акции, фонды, металлы, сырьё и валюты (TradFi) — не крипта, бот и сверка их не берут.
Тикеры некоторых совпадают с крипто-токенами на CoinGecko, и раньше они проходили фильтр по капитализации по ошибке.
NOW, NVDA, META, CRCL, SPCX, SPY проверены по архиву: объём в выходные в 7–20 раз ниже, чем в будни (у крипты — в 1–2,5).
Остальные — заведомо акции/фонды/сырьё без крипто-двойников (на случай будущих совпадений).
Тикеры, которые бывают и криптой (CAT, O, BILL, EWT, ON, NET, APP, MP и т. п.), сюда НЕ входят."""

TRADFI = {s + "USDT" for s in """
NOW NVDA META CRCL SPCX SPY
AAPL AMZN GOOGL MSFT TSLA AMD AVGO INTC NFLX ORCL PLTR MSTR COIN HOOD BABA TSM ASML PYPL UBER JPM GS BAC LLY UNH
WMT XOM MU QCOM ADBE CRM CSCO IBM DIS NKE MCD KO PFE MRK ABBV AMGN COST HD SHOP SNOW CRWD DDOG PANW MRVL KLAC LRCX
AMAT TXN SMCI DELL HPE RIVN RKLB IONQ OKLO MARA IREN HIMS SOFI RDDT PDD MELI NVO GME AMC DJT BRKB
QQQ IWM TQQQ SQQQ SOXL SOXS SMH XBI XLE GDX TZA TMF TBT UVXY URNM BITO EWJ EWY EWZ EWZS KORU KODEX200
TSLL NVDL MSTU MUU SNXX SKUU STXX MVLL CSOPSAMSUNG2L CSOPSKHYNIX2L
SAMSUNG SAMSUNGEM SKHYNIX SKHY HYUNDAI LGELECTRONICS NAVER HANMI TENCENT MEITUAN KUAISHOU HK0625 HK0700 HK0992 HK1810
OPENAI ANTHROPIC
XAU XAG XPD XPT NATGAS COPPER CL BZ USDBRL
""".split()}
