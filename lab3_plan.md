# Plan Implementacji Zadania 3: Optymalizacje Lokalnego Przeszukiwania

## Cel

Implementacja i ocena dwóch mechanizmów optymalizacji dla algorytmu lokalnego przeszukiwania w wersji stromej (steepest):
1.  Lista Ruchów (LM - Move List).
2.  Ruchy Kandydackie (Candidate Moves).

Porównanie zoptymalizowanych wersji z podstawowym lokalnym przeszukiwaniem stromym oraz najlepszą heurystyką konstrukcyjną z Zadania 1 (Weighted Regret Cycle).

## Wymagania Wstępne

*   **Algorytm Bazowy:** Local Search (Steepest, Random Initial Solution).
*   **Sąsiedztwo:** Użyjemy typu sąsiedztwa (`VertexExchange` lub `EdgeExchange`), który okazał się **lepszy** dla wersji stromej w Zadaniu 2. **Założenie:** `NeighborhoodType::EdgeExchange` był lepszy (jeśli nie, należy go zmienić w planie i kodzie).
*   **Rozwiązanie Startowe:** Zawsze losowe (`InitialSolutionType::Random`) dla wszystkich wariantów LS w tym zadaniu.
*   **Liczba Uruchomień:** 100 dla każdego algorytmu na każdej instancji.

## Kroki Implementacji

### 1. Refaktoryzacja `LocalSearch` (`src/algorithms/local_search/base.rs`)

1.1. **Zdefiniuj `OptimizationType` Enum:**
    *   Stwórz nowy enum (może być w `base.rs` lub nowym `optimization.rs`):
        ```rust
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum OptimizationType {
            None,
            CandidateMoves { k: usize }, // Parametr k dla kandydatów
            MoveList,
            // Opcjonalnie: Both { k: usize }, // Jeśli chcemy połączyć
        }
        ```
1.2. **Zaktualizuj Strukturę `LocalSearch`:**
    *   Dodaj pole `optimization_type: OptimizationType` do struktury.
1.3. **Zaktualizuj `LocalSearch::new`:**
    *   Zmodyfikuj sygnaturę, aby przyjmowała `optimization_type`.
    *   Zaktualizuj generowanie `name_str`, aby odzwierciedlało używaną optymalizację.
1.4. **Zaktualizuj `solve_with_feedback`:**
    *   W głównej pętli `loop` dodaj `match self.optimization_type { ... }`.
    *   Gałąź `OptimizationType::None` będzie zawierać dotychczasową logikę przeszukiwania stromego (generowanie wszystkich ruchów w sąsiedztwie, wybór najlepszego poprawiającego).
    *   Dodaj puste gałęzie dla `OptimizationType::CandidateMoves` i `OptimizationType::MoveList` jako placeholder.

### 2. Implementacja Optymalizacji: Ruchy Kandydackie

2.1. **Moduł dla Kandydatów:**
    *   Stwórz plik `src/algorithms/local_search/candidate_moves.rs`.
    *   Dodaj `mod candidate_moves;` w `src/algorithms/local_search/mod.rs`.
2.2. **Obliczanie Krawędzi Kandydackich:**
    *   Zdefiniuj funkcję (np. w `src/utils.rs` lub nowym module `src/utils/candidate_utils.rs`) `calculate_candidate_set(instance: &TsplibInstance, k: usize) -> HashSet<(usize, usize)>`.
    *   Funkcja ta dla każdego wierzchołka `v` znajduje `k` najbliższych sąsiadów i dodaje krawędzie `(v, neighbor)` (w formie uporządkowanej, np. `(min(v, neighbor), max(v, neighbor))`) do `HashSet`.
    *   Metoda `TsplibInstance::distance` będzie potrzebna.
2.3. **Filtrowanie Ruchów:**
    *   W `candidate_moves.rs` zdefiniuj funkcję `get_best_candidate_move(possible_moves: Vec<EvaluatedMove>, candidate_set: &HashSet<(usize, usize)>, solution: &Solution, instance: &TsplibInstance) -> Option<EvaluatedMove>`.
    *   Ta funkcja iteruje przez `possible_moves`. Dla każdego ruchu:
        *   Określ, jakie krawędzie są *dodawane* do rozwiązania przez ten ruch.
        *   Sprawdź, czy *co najmniej jedna* z dodawanych krawędzi (lub jej odwrócona wersja, jeśli przechowujemy w uporządkowany sposób) znajduje się w `candidate_set`.
        *   Filtruj ruchy, zachowując tylko te, które wprowadzają krawędź kandydacką i są poprawiające (`delta < 0`).
        *   Zwróć najlepszy (najmniejsza `delta`) spośród przefiltrowanych ruchów.
2.4. **Integracja w `base.rs`:**
    *   W gałęzi `OptimizationType::CandidateMoves { k }` w `solve_with_feedback`:
        *   Na początku (przed pętlą `loop`) oblicz `candidate_set` używając `calculate_candidate_set`.
        *   Wewnątrz pętli `loop`:
            *   Wygeneruj *wszystkie* `possible_moves` w sąsiedztwie (tak jak w `OptimizationType::None`).
            *   Wywołaj `get_best_candidate_move`, przekazując `possible_moves` i `candidate_set`.
            *   Jeśli zwrócony zostanie `Some(best_move)`, zastosuj go. W przeciwnym razie zakończ (brak poprawiających ruchów kandydackich).

### 3. Implementacja Optymalizacji: Lista Ruchów (LM)

3.1. **Moduł dla Listy Ruchów:**
    *   Stwórz plik `src/algorithms/local_search/move_list.rs`.
    *   Dodaj `mod move_list;` w `src/algorithms/local_search/mod.rs`.
3.2. **Struktura `StoredMove`:**
    *   W `move_list.rs` zdefiniuj strukturę lub enum `StoredMove`, która przechowa:
        *   Typ ruchu (np. kopia `crate::moves::types::Move` lub nowy, bardziej szczegółowy enum).
        *   Obliczoną deltę (`delta: i32`).
        *   Informacje potrzebne do weryfikacji aplikowalności, np.:
            *   Dla `IntraRouteEdgeExchange`: `cycle_id`, `(vi, vi_plus_1)`, `(vj, vj_plus_1)`.
            *   Dla `IntraRouteVertexExchange`: `cycle_id`, `pos1`, `pos2`, `prev1`, `v1`, `next1`, `prev2`, `v2`, `next2`.
            *   Dla `InterRouteExchange`: `pos1`, `pos2`, `prev_u`, `u`, `next_u`, `prev_v`, `v`, `next_v`.
        *   Może być konieczne dodanie `impl PartialEq, Eq, PartialOrd, Ord` dla sortowania. Sortowanie po `delta`.
3.3. **Struktura `MoveListManager`:**
    *   W `move_list.rs` zdefiniuj strukturę `MoveListManager` zawierającą np. `list: Vec<StoredMove>`.
    *   Implementuj metody:
        *   `new()`: Tworzy pusty menedżer.
        *   `add_moves(&mut self, new_potential_moves: Vec<EvaluatedMove>, solution: &Solution, instance: &TsplibInstance)`: Przekształca `EvaluatedMove` na `StoredMove` (dodając potrzebne informacje kontekstowe z `solution`), filtruje poprawiające (`delta < 0`), dodaje do `self.list`, usuwa duplikaty (jeśli to możliwe) i sortuje listę (`self.list.sort_unstable()`). **Uwaga:** Zadanie 3 wspomina o dodawaniu ruchów z odwróconym kierunkiem dla krawędzi - to trzeba uwzględnić przy generowaniu `new_potential_moves`.
        *   `find_best_applicable(&mut self, solution: &Solution, instance: &TsplibInstance) -> Option<StoredMove>`: Iteruje przez `self.list` od najlepszego (`delta`):
            *   Dla każdego `stored_move` sprawdza, czy jest aplikowalny do *aktualnego* `solution` (czy usuwane krawędzie/sąsiedztwa istnieją).
            *   Jeśli nieaplikowalny (np. krawędź nie istnieje) -> usuń go z `self.list`.
            *   Jeśli nieaplikowalny (np. zły kierunek krawędzi dla 2-opt) -> pozostaw go, idź dalej.
            *   Jeśli aplikowalny -> zwróć `Some(stored_move.clone())` i usuń go z `self.list`.
            *   Jeśli przejrzałeś całą listę i nic nie znalazłeś -> zwróć `None`.
3.4. **Metody Pomocnicze (Opcjonalnie):**
    *   Rozważ dodanie metod do `Solution` (np. `has_edge(u: usize, v: usize) -> bool`, `get_neighbors(v: usize, cycle_id: CycleId) -> Option<(usize, usize)>`) dla ułatwienia sprawdzania aplikowalności.
3.5. **Integracja w `base.rs`:**
    *   W gałęzi `OptimizationType::MoveList` w `solve_with_feedback`:
        *   Zainicjuj `move_manager = MoveListManager::new()`.
        *   Zainicjuj flagę `recalculate_moves = true`.
        *   Wewnątrz pętli `loop`:
            *   **Jeśli** `recalculate_moves`:
                *   Wygeneruj *wszystkie* potencjalne ruchy (w tym te "odwrócone" dla 2-opt).
                *   Wywołaj `move_manager.add_moves(...)`.
                *   Ustaw `recalculate_moves = false`.
            *   Wywołaj `move_manager.find_best_applicable(...)`.
            *   **Jeśli** `Some(best_applicable_move)`:
                *   Zastosuj ruch do `current_solution`.
                *   Zaktualizuj `current_cost`.
                *   Ustaw `recalculate_moves = true` (po zastosowaniu ruchu, oceny mogły się zmienić).
            *   **Jeśli** `None`:
                *   Zakończ pętlę (brak aplikowalnych ruchów poprawiających na liście).

### 4. Aktualizacja `main.rs`

4.1. **Instancje Algorytmów:**
    *   Stwórz instancje `LocalSearch` dla:
        *   `OptimizationType::None` (z `NeighborhoodType::EdgeExchange`, `InitialSolutionType::Random`).
        *   `OptimizationType::CandidateMoves { k: 10 }` (parametr `k` do ewentualnej zmiany; reszta jak wyżej).
        *   `OptimizationType::MoveList` (reszta jak wyżej).
    *   Dodaj instancję `WeightedRegretCycle::default()`.
4.2. **Uruchomienie Eksperymentu:**
    *   Upewnij się, że `run_experiment` jest wywoływane z `num_runs = 100`.
4.3. **Zbieranie i Prezentacja Wyników:**
    *   Zachowaj istniejącą logikę zbierania statystyk i drukowania tabeli podsumowującej.

### 5. Testowanie i Weryfikacja

*   Dokładnie przetestuj każdą optymalizację.
*   Sprawdź poprawność działania (np. czy koszty maleją, czy rozwiązania są prawidłowe).
*   Porównaj czasy wykonania i jakość rozwiązań (min/max/avg cost) między różnymi wariantami.
*   Wygeneruj wizualizacje najlepszych znalezionych rozwiązań.

---

Ten plan stanowi szczegółową mapę drogową. Możemy teraz zacząć implementację krok po kroku, zaczynając od refaktoryzacji `LocalSearch`.
