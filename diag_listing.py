"""ВРЕМЕННО (удалить): месяц появления фьючерса на Binance — чтобы отличить TradFi (2026) от старых крипто-монет."""
import re
import requests
from concurrent.futures import ThreadPoolExecutor

SYMS = """BILL CRCL EWT F GPS META NOW NVDA ON O OPEN SPCX SPY US Q T C A CVX DIA EDU ARK AR CHIP BR B MET ME LA S W G H
SPACE STAR POWER SAFE TAKE TA AT ACT ALT KAT YB EDGE ESP RE COW CAT HOME BIO GAS CAP SENT MEGA LIGHT SUN SUPER
TEAM NET ARM APP MP BE CC DRAM EIDO EQT FLNC FWDI GPRO INTW INX KORU NOM PAYP QNTX SKDD SLX STM TBT TMF TWST USAR
VKTX WET AAOI AGPU AIGENSYN BLUAI CRML BWET SNXX STXX SKUU MVLL MUU MSTU NVDL TSLL LITE BITO HIVE NOK SONY BYD
HANA HANMI CBRS ALL AIA M MON TST TUT OPG SHELL STABLE""".split()
URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/{}USDT/1d/"


def first(sym):
    try:
        x = requests.get(URL.format(sym), timeout=30).text
        months = sorted(re.findall(r"1d-(\d{4}-\d{2})\.zip<", x))
        return sym, months[0] if months else "нет помесячных (новая)"
    except Exception as e:  # noqa: BLE001
        return sym, f"ошибка {e}"


with ThreadPoolExecutor(16) as ex:
    for s, m in sorted(ex.map(first, SYMS), key=lambda r: r[1]):
        print(f"{s}: {m}")
