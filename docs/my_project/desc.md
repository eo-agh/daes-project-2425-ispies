## Cel
Głównym celem projektu jest przeprowadzenie zaawansowanej analizy predykcyjnej dotyczącej zanieczyszczenia powietrza na podstawie danych z GIOS (Głównego Inspektoratu Ochrony Środowiska). Badanie obejmuje prognozowanie stężeń różnych substancji, takich jak PM10, PM2.5, SO2 czy O3, z wykorzystaniem dwóch podejść:
- Per-sensor – indywidualne modelowanie dla każdego sensora, 
- Zgeneralizowane – modelowanie grupowe uwzględniające podobieństwo sensorów pod względem trendów czasowych lub lokalizacji przestrzennej.

Dodatkowo, projekt zakłada:
- Automatyzację procesu predykcji dla różnych zmiennych środowiskowych,
- Porównanie metod tworzenia map przestrzennych na podstawie danych aktualnych, historycznych (backtest) oraz prognozowanych,
- Wyznaczenie przedziałów predykcyjnych (prediction intervals) dla prognoz, aby uwzględnić pesymistyczne i optymistyczne scenariusze.

## Metody

W ramach projektu zostaną zastosowane następujące metody analityczne:

- Modelowanie:
    - Przeprowadzenie backtestu oraz predykcji do przodu (forecasting) z uwzględnieniem odpowiedniego horyzontu czasowego,
- Hiperoptymalizacja:
    - modeli w celu poprawy dokładności prognoz.
- Klasteryzacja:
    - Badanie efektywności modelowania indywidualnego (per-sensor) oraz grupowego,
    - Grupowanie sensorów na podstawie zmienności danych, trendów czasowych lub geolokalizacji.
- Interpolacja przestrzenna:
    - Generowanie map zanieczyszczeń na podstawie danych pomiarowych, backtestowych i prognozowanych,

- Ewaluacja jakości map poprzez leave-one-out resampling – wykluczanie kolejnych punktów pomiarowych i weryfikacja dokładności interpolacji.

- AutoML i zaawansowane techniki predykcyjne:
     - Wykorzystanie automatycznego uczenia maszynowego (AutoML) do uogólnienia systemu predykcji na różne zmienne i lokalizacje,
     - Eksperymentalne zastosowanie sekwencyjnej symulacji gaussowskiej do tworzenia map wynikowych.

## Techniczna implementacja
Projekt zostanie zrealizowany z wykorzystaniem następujących narzędzi i technologii:

- Nixtla – do budowy modeli predykcyjnych oraz klasteryzacji,
- SKtime (opcjonalnie) – jako alternatywne rozwiązanie do analizy szeregów czasowych,
- Multiprocessing – w celu przyspieszenia obliczeń poprzez równoległe przetwarzanie,
- Algorytmy genetyczne – potencjalne zastosowanie w optymalizacji hiperparametrów,
- GeoPandas – do przetwarzania i wizualizacji danych przestrzennych.


