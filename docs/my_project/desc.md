## Cel
Zbadanie danych dotyczących zanieczyszczenia powietrza z GIOSiu. Przeprowadzanie analizy predykcji na podstawie podejścia per-sensor oraz zgeneralizowanego dla grupy sensorów. Automatyzacja procesu dla przewidywania różnych zmiennych takich jak PM10, PM2.5, SO2, O3 etc. Porównanie metod tworzenia map przestrzennych na podstawie danych aktualnych, backtestowych i forecastowanych poprzez ich ewaluację. Stworzenie prediction intervals dla prognoz w celu zobrazowania pesymistycznych i optymistycznych scenariuszy.

## Metody
- Modelowanie - predykcje backtestowe oraz do przodu z uwzględniem odpowiedniego forecasting horizon dla czasu egzekucji oraz hiperotymalizacji modelu
- Klasteryzacja - zbadanie modelowania osobno dla każdego sensoru oraz dla grupy sensorów ze wględu na zmienność i trendy w danych, i/lub geolokalizację przestrzenną
- Interpolacja - stworzenie map wynikowych
- Sekwencyjna symulacja gaussowska? - stworzenie map wynikowych
- AutoML - generalizacja na różnego rodzaju zmienne dla różnych punktów (uogólnienie caego systemu)
- Leave-one-out resampling - metoda ewaluacji tworzonych map poprzez odpowiednie wykluaczanie kolejnych punktów pomiarowych i tworzenie ich na podstawie pozostalych punktów.

## Techniczna implementacja
- Nixtla (modele) (klasteryzcja?)
- SKtime (opcjonalnie) (klasteryzacja)
- Multiprocessing
- Algorytmy genetyczne?
- Geopandas

## ...
