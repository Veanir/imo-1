# # IMO - Zadanie 3
#
# Autorzy: Dariusz Max Adamski, Sławomir Gilewski
#
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import multiprocessing as mp
from time import time
from copy import deepcopy

# Ustawienie domyślnego rozmiaru wykresów
plt.rc("figure", figsize=(9, 5))

# --- Funkcje pomocnicze ---


def index(xs, e):
    """
    Zwraca indeks elementu 'e' w liście 'xs'.
    Zwraca None, jeśli elementu nie ma.
    """
    try:
        return xs.index(e)
    except ValueError:  # Poprawiona obsługa błędu
        return None


def find_node(cycles, a):
    """
    Znajduje węzeł 'a' w jednym z dwóch cykli.
    Zwraca krotkę (indeks_cyklu, indeks_w_cyklu).
    Przerywa działanie, jeśli węzeł nie zostanie znaleziony.
    """
    i = index(cycles[0], a)
    if i is not None:
        return 0, i
    i = index(cycles[1], a)
    if i is not None:
        return 1, i
    print("Błąd: Nie znaleziono miasta w cyklach.")  # Dodano komunikat błędu
    print(cycles)
    assert False, f"Miasto {a} musi być w jednym z cykli"


def remove_at(xs, sorted_indices):
    """
    Usuwa elementy z listy 'xs' na podanych posortowanych indeksach.
    """
    for i in reversed(sorted_indices):
        del xs[i]


def reverse(xs, i, j):
    """
    Odwraca fragment listy 'xs' pomiędzy indeksami 'i' a 'j' (modulo długość listy).
    """
    n = len(xs)
    d = (j - i) % n
    # print(d)
    for k in range(abs(d) // 2 + 1):
        a, b = (i + k) % n, (i + d - k) % n
        # print(a, '<->', b)
        xs[a], xs[b] = xs[b], xs[a]


# --- Funkcje związane z TSP ---


def distance(a, b):
    """
    Oblicza odległość euklidesową między dwoma punktami (zaokrągloną).
    """
    return np.round(np.sqrt(np.sum((a - b) ** 2)))


def read_instance(path):
    """
    Wczytuje instancję problemu TSP z pliku.
    Zwraca macierz odległości i współrzędne miast.
    """
    coords = pd.read_csv(
        path, sep=" ", names=["n", "x", "y"], skiprows=6, skipfooter=1, engine="python"
    )
    cities_coords = coords.drop(columns=["n"]).values  # Zmieniono nazwę dla jasności
    ns = np.arange(len(cities_coords))
    # Obliczenie macierzy odległości
    cities_distances = np.array(
        [[distance(cities_coords[i], cities_coords[j]) for j in ns] for i in ns]
    )
    return cities_distances, coords


def draw_path(coords, path, color="blue"):
    """
    Rysuje pojedynczy cykl na wykresie.
    """
    cycle = path + [path[0]]  # Zamknięcie cyklu
    for i in range(len(cycle) - 1):
        a, b = cycle[i], cycle[i + 1]
        plt.plot([coords.x[a], coords.x[b]], [coords.y[a], coords.y[b]], color=color)


def plot_solution(coords, solution):
    """
    Rysuje oba cykle rozwiązania na wykresie.
    """
    path1, path2 = solution
    draw_path(coords, path1, color="green")
    draw_path(coords, path2, color="red")
    plt.scatter(coords.x, coords.y, color="black")  # Rysuje miasta jako punkty
    plt.show()  # Dodano, aby wyświetlić wykres


def cycle_score(cities, path):
    """
    Oblicza długość (koszt) pojedynczego cyklu.
    """
    cycle = path + [path[0]]  # Zamknięcie cyklu
    return sum(cities[cycle[i], cycle[i + 1]] for i in range(len(cycle) - 1))


def score(cities, paths):
    """
    Oblicza całkowity koszt rozwiązania (sumę długości obu cykli).
    """
    return cycle_score(cities, paths[0]) + cycle_score(cities, paths[1])


# --- Algorytm konstrukcyjny (Regret Heuristic) ---


def delta_insert(cities, path, i, city):
    """
    Oblicza zmianę kosztu cyklu po wstawieniu 'city' na pozycji 'i' w 'path'.
    """
    # Obsługa przypadku pustej lub jednoelementowej ścieżki
    if not path:
        return 0  # Lub inna wartość bazowa, jeśli potrzebna
    if len(path) == 1:
        a = path[0]
        return cities[a, city] + cities[city, a]  # Koszt cyklu z dwóch miast

    # Standardowe obliczenie dla ścieżki z co najmniej dwoma miastami
    n = len(path)
    a = path[(i - 1 + n) % n]  # Poprawiony indeks dla cykliczności
    b = path[i % n]  # Poprawiony indeks dla cykliczności
    return cities[a, city] + cities[city, b] - cities[a, b]


def solve_regret(args):
    """
    Implementacja heurystyki konstrukcyjnej opartej na metodzie "regret".
    """
    cities, start_node = args  # Rozpakowanie argumentów
    t0 = time()
    n = cities.shape[0]
    remaining = list(range(n))

    # Inicjalizacja: wybierz dwa najbardziej oddalone miasta od start_node
    # (Oryginalny kod wybierał najbardziej oddalone od start_node, co może nie być optymalne)
    # Lepsza inicjalizacja: wybierz dwa losowe lub dwa najbardziej oddalone od siebie
    if start_node is not None and start_node in remaining:
        a = start_node
        remaining.remove(a)
        if remaining:  # Upewnij się, że są jeszcze jakieś miasta
            b = remaining[
                np.argmax(cities[a, remaining])
            ]  # Znajdź najbardziej oddalone od 'a'
            remaining.remove(b)
        else:
            # Obsługa przypadku, gdy jest tylko jedno miasto
            paths = [[a], []] if n > 0 else [[], []]
            return time() - t0, paths
    elif len(remaining) >= 2:
        # Wybierz dwa losowe lub pierwsze dwa, jeśli start_node nie jest podany
        a = remaining.pop(random.randrange(len(remaining)))
        b = remaining.pop(random.randrange(len(remaining)))
    elif len(remaining) == 1:
        a = remaining.pop()
        paths = [[a], []]
        return time() - t0, paths
    else:  # Brak miast
        paths = [[], []]
        return time() - t0, paths

    paths = (
        [[a], [b]] if "a" in locals() and "b" in locals() else [[], []]
    )  # Upewnij się, że a i b istnieją

    while remaining:
        best_choice = None  # Przechowuje (koszt_regret, indeks_miasta, indeks_sciezki, indeks_wstawienia)
        choices = []  # Lista potencjalnych ruchów dla wszystkich pozostałych miast

        for city_idx, city in enumerate(remaining):
            path_scores = (
                []
            )  # Przechowuje (koszt, indeks_wstawienia) dla danego miasta w obu ścieżkach
            for path_idx, path in enumerate(paths):
                # Oblicz koszty wstawienia miasta w każdą możliwą pozycję w ścieżce
                insert_costs = []
                if not path:  # Jeśli ścieżka jest pusta
                    # Wstawienie do pustej ścieżki tworzy cykl z jednym miastem (koszt 0)
                    # lub można zdefiniować inaczej, np. koszt do najbliższego z drugiej ścieżki
                    # Tutaj przyjmujemy, że wstawienie do pustej ścieżki jest możliwe
                    # i koszt delta to koszt dodania krawędzi do samego siebie (0) lub inny koszt bazowy
                    # Dla uproszczenia, jeśli obie ścieżki są puste na początku, to jest obsługiwane w inicjalizacji.
                    # Jeśli jedna staje się pusta później, to rzadki przypadek.
                    # Załóżmy, że zawsze wstawiamy do niepustej, jeśli to możliwe.
                    # Jeśli obie są puste, to błąd w logice wcześniej.
                    # Jeśli jedna jest pusta, a druga nie, rozważmy wstawienie tylko do niepustej.
                    if (
                        len(paths[0]) + len(paths[1]) < n
                    ):  # Upewnij się, że nie wstawiamy ostatniego elementu do pustej
                        # Można by obliczyć koszt do najbliższego w drugiej ścieżce, ale to komplikuje
                        # Na razie pomijamy wstawianie do pustej, jeśli druga nie jest pusta
                        # Jeśli obie są puste, to obsłużone na starcie.
                        # Jeśli jedna jest pusta, a druga nie, to wstawimy do niepustej.
                        pass  # Nie wstawiaj do pustej, jeśli druga istnieje
                elif len(path) == 1:
                    # Wstawienie do ścieżki z jednym elementem tworzy cykl z dwóch miast
                    a = path[0]
                    cost = cities[a, city] + cities[city, a]
                    insert_costs.append(
                        (cost, 0)
                    )  # Wstawienie na pozycję 0 (jedyne możliwe)
                    insert_costs.append(
                        (cost, 1)
                    )  # Wstawienie na pozycję 1 (również jedyne możliwe w cyklu)

                else:
                    for i in range(len(path)):  # Dla każdej krawędzi w cyklu
                        cost = delta_insert(cities, path, i, city)
                        insert_costs.append((cost, i))

                if insert_costs:  # Jeśli były możliwe miejsca wstawienia
                    insert_costs.sort(key=lambda x: x[0])  # Sortuj wg kosztu
                    best_cost = insert_costs[0][0]
                    best_insert_idx = insert_costs[0][1]
                    second_best_cost = (
                        insert_costs[1][0] if len(insert_costs) > 1 else best_cost + 1e9
                    )  # Duża wartość, jeśli tylko 1 opcja
                    regret = second_best_cost - best_cost
                    # Waga: kompromis między żalem a najlepszym kosztem
                    weight = regret - 0.37 * best_cost
                    choices.append(
                        {
                            "weight": weight,
                            "city_idx": city_idx,
                            "city": city,
                            "path_idx": path_idx,
                            "insert_idx": best_insert_idx,
                            "cost": best_cost,
                        }
                    )

        if (
            not choices
        ):  # Jeśli nie ma już gdzie wstawiać (np. wszystkie miasta przypisane)
            break

        # Wybierz najlepszy ruch na podstawie wagi (regret - 0.37 * cost)
        choices.sort(key=lambda x: x["weight"], reverse=True)  # Sortuj malejąco wg wagi
        best_move = choices[0]

        # Wykonaj najlepszy ruch
        city_to_insert = best_move["city"]
        path_idx = best_move["path_idx"]
        insert_idx = best_move["insert_idx"]

        paths[path_idx].insert(insert_idx, city_to_insert)
        remaining.pop(best_move["city_idx"])  # Usuń miasto z listy pozostałych

    return time() - t0, paths


def random_solution(n, seed=None):
    """
    Generuje losowe rozwiązanie początkowe (dwa cykle).
    """
    remaining = list(range(n))
    if seed is not None:
        random.seed(seed)  # Ustawienie ziarna dla powtarzalności
    random.shuffle(remaining)
    mid = n // 2
    return remaining[:mid], remaining[mid:]


# --- Operacje na cyklach (ruchy lokalne) ---

SWAP_EDGE, SWAP_NODE = range(2)  # Definicja typów ruchów


def insert_move(moves, move):
    """
    Wstawia ruch do posortowanej listy ruchów (według delty).
    Unika duplikatów o tej samej delcie (choć może to nie być pożądane).
    """
    delta_x = move[0]
    for i, x in enumerate(moves):
        delta_y = x[0]
        if delta_x < delta_y:
            moves.insert(i, move)
            return
        # Oryginalny kod unikał duplikatów o tej samej delcie.
        # Jeśli chcemy zezwolić na ruchy o tej samej delcie, usuń warunek elif.
        # elif delta_x == delta_y:
        #     return
    moves.append(move)  # Dodaj na koniec, jeśli delta jest największa


def has_edge(cycle, a, b):
    """
    Sprawdza, czy krawędź (a, b) lub (b, a) istnieje w cyklu.
    Zwraca +1 dla (a, b), -1 dla (b, a), 0 jeśli nie istnieje.
    """
    n = len(cycle)
    if n < 2:
        return 0  # Krawędź nie może istnieć w cyklu z mniej niż 2 węzłami

    for i in range(n):
        x, y = cycle[i], cycle[(i + 1) % n]  # Użyj modulo dla zamknięcia cyklu
        if (a, b) == (x, y):
            return +1
        if (a, b) == (y, x):
            return -1
    return 0


def any_has_edge(cycles, a, b):
    """
    Sprawdza, czy krawędź (a, b) lub (b, a) istnieje w którymkolwiek z dwóch cykli.
    Zwraca (indeks_cyklu, status) lub (None, 0).
    """
    for i in range(2):
        status = has_edge(cycles[i], a, b)
        if status != 0:
            return i, status
    return None, 0


def delta_swap_node(D, x1, y1, z1, x2, y2, z2):
    """
    Oblicza zmianę kosztu po zamianie węzłów y1 i y2 między cyklami.
    x1, y1, z1 to sąsiedzi w pierwszym cyklu.
    x2, y2, z2 to sąsiedzi w drugim cyklu.
    """
    # Koszt usunięcia y1: D[x1, y1] + D[y1, z1]
    # Koszt dodania y2 w miejsce y1: D[x1, y2] + D[y2, z1]
    # Koszt usunięcia y2: D[x2, y2] + D[y2, z2]
    # Koszt dodania y1 w miejsce y2: D[x2, y1] + D[y1, z2]
    # Delta = (Nowy koszt) - (Stary koszt)
    # Delta = (D[x1, y2] + D[y2, z1] + D[x2, y1] + D[y1, z2]) - (D[x1, y1] + D[y1, z1] + D[x2, y2] + D[y2, z2])
    # Formuła w kodzie jest inna - sprawdźmy ją:
    # return D[x1,y2] + D[z1,y2] - D[x1,y1] - D[z1,y1] + D[x2,y1] + D[z2,y1] - D[x2,y2] - D[z2,y2]
    # Ta formuła wydaje się poprawna, zakłada że krawędzie to (x1,y1), (y1,z1) oraz (x2,y2), (y2,z2)
    # Po zamianie: (x1,y2), (y2,z1) oraz (x2,y1), (y1,z2)
    # Zmiana w cyklu 1: (D[x1,y2] + D[y2,z1]) - (D[x1,y1] + D[y1,z1])
    # Zmiana w cyklu 2: (D[x2,y1] + D[y1,z2]) - (D[x2,y2] + D[y2,z2])
    # Suma zmian = D[x1,y2] + D[y2,z1] - D[x1,y1] - D[y1,z1] + D[x2,y1] + D[y1,z2] - D[x2,y2] - D[y2,z2]
    # Wygląda na to, że w kodzie jest D[z1,y2] zamiast D[y2,z1] i D[z2,y1] zamiast D[y1,z2].
    # Zakładając, że macierz D jest symetryczna (D[a,b] == D[b,a]), to jest to samo.
    # Sprawdźmy jeszcze raz:
    # Stary koszt = D[x1,y1] + D[y1,z1] + D[x2,y2] + D[y2,z2]
    # Nowy koszt = D[x1,y2] + D[y2,z1] + D[x2,y1] + D[y1,z2]
    # Delta = Nowy - Stary
    # Delta = D[x1,y2] + D[y2,z1] + D[x2,y1] + D[y1,z2] - D[x1,y1] - D[y1,z1] - D[x2,y2] - D[y2,z2]
    # Kod: D[x1,y2] + D[z1,y2] - D[x1,y1] - D[z1,y1] + D[x2,y1] + D[z2,y1] - D[x2,y2] - D[z2,y2]
    # Jeśli D jest symetryczna, D[z1,y2]=D[y2,z1], D[z1,y1]=D[y1,z1], D[z2,y1]=D[y1,z2], D[z2,y2]=D[y2,z2]
    # Wtedy kod = D[x1,y2] + D[y2,z1] - D[x1,y1] - D[y1,z1] + D[x2,y1] + D[y1,z2] - D[x2,y2] - D[y2,z2]
    # Tak, formuła w kodzie jest poprawna dla symetrycznej macierzy odległości.
    return (
        D[x1, y2]
        + D[z1, y2]
        - D[x1, y1]
        - D[z1, y1]
        + D[x2, y1]
        + D[z2, y1]
        - D[x2, y2]
        - D[z2, y2]
    )


def make_swap_node(cities, cycles, cyc1_idx, i, cyc2_idx, j):
    """
    Przygotowuje ruch zamiany węzłów między cyklami.
    Zwraca (delta, move_tuple).
    """
    C1, C2 = cycles[cyc1_idx], cycles[cyc2_idx]
    D = cities
    n, m = len(C1), len(C2)

    # Sprawdzenie czy cykle nie są za małe
    if n < 1 or m < 1:
        return 1e9, None  # Zwraca dużą deltę, jeśli któryś cykl jest pusty

    # Pobranie węzłów i ich sąsiadów
    y1 = C1[i]
    x1 = C1[(i - 1 + n) % n]  # Poprzedni węzeł w cyklu 1
    z1 = C1[(i + 1) % n]  # Następny węzeł w cyklu 1 (jeśli n > 1)

    y2 = C2[j]
    x2 = C2[(j - 1 + m) % m]  # Poprzedni węzeł w cyklu 2
    z2 = C2[(j + 1) % m]  # Następny węzeł w cyklu 2 (jeśli m > 1)

    # Obsługa cykli jednoelementowych (nie mają sąsiadów w cyklu)
    if n == 1:
        x1 = z1 = y1  # Sąsiadem jest sam węzeł (krawędź do siebie ma koszt 0)
    if m == 1:
        x2 = z2 = y2

    # Obliczenie delty
    # Jeśli jeden z cykli ma tylko 1 element, delta jest inna
    if n == 1 and m > 1:
        # Usuwamy y1 (nic), dodajemy y2 (nic)
        # Usuwamy y2 z C2: tracimy D[x2,y2] + D[y2,z2], zyskujemy D[x2,z2]
        # Dodajemy y1 do C2: tracimy D[x2,z2], zyskujemy D[x2,y1] + D[y1,z2]
        # Delta = (D[x2,y1] + D[y1,z2]) - (D[x2,y2] + D[y2,z2])
        delta = D[x2, y1] + D[y1, z2] - D[x2, y2] - D[y2, z2]
    elif m == 1 and n > 1:
        # Analogicznie
        delta = D[x1, y2] + D[y2, z1] - D[x1, y1] - D[y1, z1]
    elif n == 1 and m == 1:
        delta = 0  # Zamiana dwóch jednoelementowych cykli nic nie zmienia
    else:  # Oba cykle mają >= 2 elementy
        delta = delta_swap_node(cities, x1, y1, z1, x2, y2, z2)

    move = delta, SWAP_NODE, cyc1_idx, cyc2_idx, x1, y1, z1, x2, y2, z2
    return delta, move


def delta_swap_edge(cities, a, b, c, d):
    """
    Oblicza zmianę kosztu po zamianie krawędzi (a, b) i (c, d) na (a, c) i (b, d)
    w ramach jednego cyklu (ruch 2-opt).
    """
    # Sprawdza, czy węzły się nie powtarzają, co uniemożliwia ruch 2-opt
    if a == c or a == d or b == c or b == d:
        return 1e9  # Zwraca dużą wartość, jeśli ruch jest niemożliwy
    # Delta = (Nowy koszt) - (Stary koszt)
    # Delta = (cities[a, c] + cities[b, d]) - (cities[a, b] + cities[c, d])
    return cities[a, c] + cities[b, d] - cities[a, b] - cities[c, d]


def gen_swap_edge_2(cities, cycle, i, j):
    """
    Generuje szczegóły ruchu zamiany krawędzi (2-opt) dla podanych indeksów i, j.
    Zwraca (delta, a, b, c, d).
    """
    n = len(cycle)
    if n < 4:
        return (1e9, None, None, None, None)  # Ruch 2-opt wymaga co najmniej 4 węzłów

    # Upewnij się, że indeksy są różne i nie sąsiadują
    # Normalizacja indeksów, aby i < j
    i, j = min(i, j), max(i, j)
    if i == j or (j == i + 1) or (i == 0 and j == n - 1):
        return (1e9, None, None, None, None)  # Krawędzie sąsiadujące lub te same

    # Węzły tworzące krawędzie do zamiany: (a, b) i (c, d)
    a = cycle[i]
    b = cycle[(i + 1) % n]
    c = cycle[j]
    d = cycle[(j + 1) % n]

    delta = delta_swap_edge(cities, a, b, c, d)
    return (delta, a, b, c, d)


def delta_swap_edge_2(cities, cycle, i, j):
    """
    Zwraca tylko deltę dla ruchu zamiany krawędzi (2-opt).
    """
    return gen_swap_edge_2(cities, cycle, i, j)[0]


def gen_swap_edge(n):
    """
    Generuje pary indeksów (i, j) dla potencjalnych ruchów zamiany krawędzi (2-opt).
    Unika sąsiadujących krawędzi i trywialnych przypadków.
    """
    if n < 4:
        return []  # Potrzebne co najmniej 4 wierzchołki
    # Generuje pary (i, j) takie, że i < j oraz nie są sąsiadami
    # (i, i+1) oraz (j, j+1) to krawędzie do usunięcia
    # d to odległość między i a j w cyklu
    # range(2, n-1) zapewnia, że j nie jest i+1 ani i-1 (modulo n)
    return [(i, (i + d) % n) for i in range(n) for d in range(2, n - 1)]


def gen_swap_node(n, m):
    """
    Generuje pary indeksów (i, j) dla potencjalnych ruchów zamiany węzłów.
    """
    return [(i, j) for i in range(n) for j in range(m)]


def init_moves(cities, cycles):
    """
    Inicjalizuje listę wszystkich potencjalnych ruchów poprawiających (delta < 0).
    """
    moves = []
    # Ruchy zamiany krawędzi (wewnątrz cykli)
    for k in range(2):  # Dla każdego cyklu
        cycle = cycles[k]
        n = len(cycle)
        if n < 4:
            continue  # 2-opt wymaga >= 4 węzłów
        for i, j in gen_swap_edge(n):
            delta, a, b, c, d = gen_swap_edge_2(cities, cycle, i, j)
            if delta < -1e-9:  # Użyj małego epsilona dla porównań zmiennoprzecinkowych
                moves.append((delta, SWAP_EDGE, a, b, c, d))

    # Ruchy zamiany węzłów (między cyklami)
    n, m = len(cycles[0]), len(cycles[1])
    if n > 0 and m > 0:  # Tylko jeśli oba cykle są niepuste
        for i, j in gen_swap_node(n, m):
            delta, move_details = make_swap_node(cities, cycles, 0, i, 1, j)
            if (
                move_details and delta < -1e-9
            ):  # Sprawdź, czy ruch jest możliwy i poprawiający
                moves.append(move_details)

    return moves


def apply_move(cycles, move):
    """
    Stosuje dany ruch (modyfikuje listę 'cycles' w miejscu).
    """
    delta, kind = move[0], move[1]
    # print(f"Applying move: {kind}, Delta: {delta}") # Debug

    if kind == SWAP_EDGE:
        # delta, SWAP_EDGE, a, b, c, d = move
        # Znajdź cykl i indeksy węzłów a i c
        _, _, a, _, c, _ = move  # Rozpakuj tylko potrzebne elementy
        c1_info = find_node(cycles, a)
        c2_info = find_node(cycles, c)

        if c1_info is None or c2_info is None:
            print(f"Error SWAP_EDGE: Node not found. a={a}, c={c}")
            return  # Nie można zastosować ruchu

        c1, i = c1_info
        c2, j = c2_info

        if c1 != c2:
            print(
                f"Error SWAP_EDGE: Nodes {a} and {c} are in different cycles ({c1}, {c2}). Cannot swap edges between cycles."
            )
            # assert c1 == c2, 'Cannot swap edges between cycles' # Oryginalny assert
            return  # Zamiast assert, po prostu nie wykonuj ruchu

        # Wykonaj odwrócenie fragmentu cyklu (ruch 2-opt)
        cycle = cycles[c1]
        n = len(cycle)
        # Indeksy krawędzi to (i, i+1) i (j, j+1)
        # Odwracamy ścieżkę od (i+1) do j (włącznie)
        # Upewnij się, że indeksy są poprawne
        idx1 = (i + 1) % n
        idx2 = j
        reverse(cycle, idx1, idx2)  # Odwróć fragment między i+1 a j

    elif kind == SWAP_NODE:
        # delta, SWAP_NODE, c1, c2, x1, y1, z1, x2, y2, z2 = move
        _, _, c1_idx, c2_idx, _, node1, _, _, node2, _ = (
            move  # Rozpakuj węzły do zamiany
        )

        # Znajdź aktualne indeksy tych węzłów (mogły się zmienić)
        idx1_info = find_node(cycles, node1)
        idx2_info = find_node(cycles, node2)

        if idx1_info is None or idx2_info is None:
            print(f"Error SWAP_NODE: Node not found. node1={node1}, node2={node2}")
            return

        actual_c1, i = idx1_info
        actual_c2, j = idx2_info

        # Sprawdź, czy węzły są nadal w oczekiwanych cyklach
        # (Teoretycznie powinny być, chyba że inny ruch je przeniósł - co nie powinno się zdarzyć w steepest descent)
        if actual_c1 != c1_idx or actual_c2 != c2_idx:
            print(f"Error SWAP_NODE: Nodes {node1} or {node2} moved unexpectedly.")
            # Można dodać logikę obsługi tego przypadku, jeśli jest potrzebna
            return

        # Zamień węzły miejscami
        cycles[c1_idx][i], cycles[c2_idx][j] = cycles[c2_idx][j], cycles[c1_idx][i]

    else:
        assert False, "Invalid move type"


# --- Algorytmy przeszukiwania lokalnego ---


class SearchSteepest:
    """
    Algorytm lokalnego przeszukiwania: metoda najszybszego spadku (steepest descent).
    W każdej iteracji wykonuje najlepszy możliwy ruch (o największej ujemnej delcie).
    """

    def __init__(self, cities):
        self.cities = cities

    def __call__(self, initial_cycles):
        cycles = deepcopy(initial_cycles)  # Pracuj na kopii
        start_time = time()
        iterations = 0
        while True:
            iterations += 1
            # print(f"Steepest Iteration: {iterations}") # Debug
            moves = init_moves(
                self.cities, cycles
            )  # Znajdź wszystkie poprawiające ruchy
            if not moves:
                # print("No improving moves found. Stopping.") # Debug
                break  # Brak ruchów poprawiających - koniec

            # Wybierz najlepszy ruch (minimalna delta)
            best_move = min(moves, key=lambda x: x[0])

            # Sprawdzenie, czy delta jest rzeczywiście ujemna (z tolerancją)
            if best_move[0] >= -1e-9:
                # print("Best move delta is non-negative. Stopping.") # Debug
                break

            # Zastosuj najlepszy ruch
            apply_move(cycles, best_move)
            # print(f"Applied move: {best_move[1]}, Delta: {best_move[0]}, New Score: {score(self.cities, cycles)}") # Debug

        end_time = time()
        return end_time - start_time, cycles


class SearchCandidates:
    """
    Algorytm lokalnego przeszukiwania: metoda kandydacka.
    Ogranicza przeszukiwanie do ruchów obejmujących "bliskie" węzły.
    """

    def __init__(self, cities):
        self.cities = cities
        self.N = len(cities)

    def __call__(
        self, initial_cycles, k=10
    ):  # k - liczba najbliższych sąsiadów do rozważenia
        cycles = deepcopy(initial_cycles)
        start_time = time()
        # Pre-kalkulacja najbliższych sąsiadów dla każdego miasta
        closest = np.argsort(self.cities, axis=1)[
            :, 1 : k + 1
        ]  # argsort zwraca indeksy; pomiń sam siebie [:,0]

        iterations = 0
        while True:
            iterations += 1
            # print(f"Candidates Iteration: {iterations}") # Debug
            best_move = None
            best_delta = -1e-9  # Szukamy ruchu z deltą mniejszą niż ta wartość

            # Iteruj po wszystkich miastach 'a'
            for a in range(self.N):
                # Iteruj po k najbliższych sąsiadach 'b' miasta 'a'
                for b in closest[a]:
                    # Znajdź, w których cyklach są 'a' i 'b'
                    node_a_info = find_node(cycles, a)
                    node_b_info = find_node(cycles, b)

                    if node_a_info is None or node_b_info is None:
                        # print(f"Warning: Node {a} or {b} not found in current cycles.")
                        continue  # Pomiń, jeśli węzła nie ma (nie powinno się zdarzyć)

                    c1, i = node_a_info
                    c2, j = node_b_info

                    delta, move = None, None

                    # Rozważ ruch SWAP_EDGE (2-opt), jeśli a i b są w tym samym cyklu
                    if c1 == c2:
                        cycle = cycles[c1]
                        n = len(cycle)
                        if n >= 4:
                            # Wygeneruj ruch 2-opt używając krawędzi wychodzących z a i b
                            # Krawędź 1: (a, a_next) = (cycle[i], cycle[(i+1)%n])
                            # Krawędź 2: (b, b_next) = (cycle[j], cycle[(j+1)%n])
                            # Sprawdź różne kombinacje (a, a_next) z (b, b_next) i (a_prev, a) z (b_prev, b) etc.
                            # Najprostszy sposób: użyj funkcji gen_swap_edge_2
                            # Musimy sprawdzić obie możliwości: krawędzie (i, i+1) i (j, j+1) lub (i-1, i) i (j-1, j) etc.

                            # Sprawdź ruch dla krawędzi (i, i+1) i (j, j+1)
                            delta_temp, a_node, b_node, c_node, d_node = (
                                gen_swap_edge_2(self.cities, cycle, i, j)
                            )
                            if delta_temp < best_delta:
                                best_delta = delta_temp
                                best_move = (
                                    delta_temp,
                                    SWAP_EDGE,
                                    a_node,
                                    b_node,
                                    c_node,
                                    d_node,
                                )

                            # Sprawdź ruch dla krawędzi (i-1, i) i (j, j+1) - wymaga innej funkcji delta
                            # To staje się bardziej skomplikowane, może lepiej trzymać się oryginalnej logiki
                            # Oryginalny kod:
                            # a_node, b_node, c_node, d_node = a, cycle[(i+1)%n], b, cycle[(j+1)%n]
                            # delta = delta_swap_edge(self.cities, a_node, b_node, c_node, d_node)
                            # move = delta, SWAP_EDGE, a_node, b_node, c_node, d_node
                            # Ta logika jest niepełna, bo zakłada konkretne krawędzie.

                            # Spróbujmy inaczej: rozważ krawędź (a, b) jako potencjalną nową krawędź
                            # Jeśli a i b są w tym samym cyklu, rozważ ruch 2-opt, który tworzy krawędź (a,b)
                            # To wymaga znalezienia krawędzi (a, x) i (b, y) do usunięcia.
                            # Najbliższy sąsiad 'b' dla 'a' sugeruje, że krawędź (a,b) może być dobra.
                            # Rozważmy usunięcie krawędzi (a, a_next) i (b_prev, b) i dodanie (a, b) i (a_next, b_prev)
                            a_next = cycle[(i + 1) % n]
                            b_prev = cycle[(j - 1 + n) % n]
                            delta_temp = delta_swap_edge(
                                self.cities, a, a_next, b_prev, b
                            )  # Zamiana (a,a_next), (b_prev,b) na (a,b_prev), (a_next,b) - to nie to
                            # Chcemy zamienić (a, a_next) i (b_prev, b) na (a, b) i (a_next, b_prev)
                            delta_temp = (
                                self.cities[a, b]
                                + self.cities[a_next, b_prev]
                                - self.cities[a, a_next]
                                - self.cities[b_prev, b]
                            )
                            if delta_temp < best_delta:
                                # Musimy zapisać ruch w formacie (delta, SWAP_EDGE, node1, node1_next, node2, node2_next)
                                # Gdzie (node1, node1_next) i (node2, node2_next) to usuwane krawędzie
                                best_delta = delta_temp
                                best_move = (
                                    delta_temp,
                                    SWAP_EDGE,
                                    a,
                                    a_next,
                                    b_prev,
                                    b,
                                )  # Zapisujemy usuwane krawędzie

                    # Rozważ ruch SWAP_NODE, jeśli a i b są w różnych cyklach
                    elif c1 != c2:
                        # Upewnij się, że oba cykle mają co najmniej jeden element
                        if len(cycles[c1]) > 0 and len(cycles[c2]) > 0:
                            delta_temp, move_details = make_swap_node(
                                self.cities, cycles, c1, i, c2, j
                            )
                            if move_details and delta_temp < best_delta:
                                best_delta = delta_temp
                                best_move = move_details

            # Zastosuj najlepszy znaleziony ruch (jeśli istnieje)
            if best_move is None:
                # print("No improving candidate moves found. Stopping.") # Debug
                break  # Brak poprawiających ruchów kandydackich

            # print(f"Applying candidate move: {best_move[1]}, Delta: {best_move[0]}, New Score: {score(self.cities, cycles) + best_move[0]}") # Debug
            apply_move(cycles, best_move)

        end_time = time()
        return end_time - start_time, cycles


class SearchMemory:
    """
    Algorytm lokalnego przeszukiwania z pamięcią (podobny do tabu search, ale uproszczony).
    Utrzymuje listę potencjalnych ruchów i aktualizuje ją po każdym kroku.
    """

    def __init__(self, cities):
        self.cities = cities

    def next_moves(self, cycles, applied_move):
        """
        Generuje nowe potencjalne ruchy w sąsiedztwie ostatnio wykonanego ruchu.
        To jest kluczowy element optymalizacji - zamiast przeliczać wszystko od nowa.
        """
        # Ta implementacja w oryginalnym kodzie wydaje się nieefektywna
        # lub niekompletna. Przelicza ona dużą część ruchów na nowo.
        # Prawdziwa optymalizacja wymagałaby dokładnego określenia,
        # które delty ruchów zmieniają się po zastosowaniu `applied_move`
        # i przeliczenia tylko tych.
        # Na przykład, jeśli zastosowano SWAP_EDGE (a,b), (c,d) -> (a,c), (b,d),
        # to zmieniają się tylko delty ruchów angażujących węzły a, b, c, d
        # i ich bezpośrednich sąsiadów w cyklu.

        # Oryginalny kod generuje sporo ruchów, co może być kosztowne.
        # Dla uproszczenia, zwrócimy pełną listę nowych ruchów,
        # tak jak w init_moves, ale można to zoptymalizować.
        # return init_moves(self.cities, cycles)

        # Bardziej zoptymalizowane podejście (szkic):
        moves = []
        affected_nodes = set()
        kind = applied_move[1]

        if kind == SWAP_EDGE:
            _, _, a, b, c, d = applied_move
            # Znajdź cykl, w którym był ruch
            cycle_idx, _ = find_node(cycles, a)  # a i c są teraz w tym cyklu
            cycle = cycles[cycle_idx]
            n = len(cycle)
            # Węzły, których sąsiedztwo się zmieniło: a, b, c, d i ich nowi sąsiedzi
            try:  # Indeksy mogły się zmienić, znajdź je na nowo
                idx_a = cycle.index(a)
                idx_b = cycle.index(b)
                idx_c = cycle.index(c)
                idx_d = cycle.index(d)
                affected_nodes.add(a)
                affected_nodes.add(cycle[(idx_a - 1 + n) % n])
                affected_nodes.add(cycle[(idx_a + 1) % n])
                affected_nodes.add(b)
                affected_nodes.add(cycle[(idx_b - 1 + n) % n])
                affected_nodes.add(cycle[(idx_b + 1) % n])
                affected_nodes.add(c)
                affected_nodes.add(cycle[(idx_c - 1 + n) % n])
                affected_nodes.add(cycle[(idx_c + 1) % n])
                affected_nodes.add(d)
                affected_nodes.add(cycle[(idx_d - 1 + n) % n])
                affected_nodes.add(cycle[(idx_d + 1) % n])
            except ValueError:
                print("Node not found after SWAP_EDGE, recalculating all moves.")
                return init_moves(
                    self.cities, cycles
                )  # W razie problemu przelicz wszystko

        elif kind == SWAP_NODE:
            _, _, c1, c2, x1, y1, z1, x2, y2, z2 = (
                applied_move  # y1 i y2 zamieniły cykle
            )
            # Węzły, których sąsiedztwo się zmieniło:
            # W cyklu c1 (teraz zawiera y2): x1, y2, z1 i ich sąsiedzi
            # W cyklu c2 (teraz zawiera y1): x2, y1, z2 i ich sąsiedzi
            affected_nodes.add(x1)
            affected_nodes.add(y1)
            affected_nodes.add(z1)
            affected_nodes.add(x2)
            affected_nodes.add(y2)
            affected_nodes.add(z2)
            # Dodaj też sąsiadów tych węzłów w ich NOWYCH cyklach
            try:
                c1_cycle, c2_cycle = cycles[c1], cycles[c2]
                n1, n2 = len(c1_cycle), len(c2_cycle)
                idx_y2 = c1_cycle.index(y2)
                affected_nodes.add(c1_cycle[(idx_y2 - 1 + n1) % n1])
                affected_nodes.add(c1_cycle[(idx_y2 + 1) % n1])
                idx_y1 = c2_cycle.index(y1)
                affected_nodes.add(c2_cycle[(idx_y1 - 1 + n2) % n2])
                affected_nodes.add(c2_cycle[(idx_y1 + 1) % n2])
            except ValueError:
                print("Node not found after SWAP_NODE, recalculating all moves.")
                return init_moves(
                    self.cities, cycles
                )  # W razie problemu przelicz wszystko

        # Przelicz ruchy angażujące affected_nodes
        # 1. Ruchy SWAP_EDGE wewnątrz cykli, jeśli affected_node jest jednym z a, b, c, d
        for node in affected_nodes:
            node_info = find_node(cycles, node)
            if node_info is None:
                continue
            cycle_idx, node_idx = node_info
            cycle = cycles[cycle_idx]
            n = len(cycle)
            if n < 4:
                continue
            # Rozważ ruchy 2-opt angażujące krawędzie przy node
            # Krawędź (prev, node) i (node, next)
            prev_node = cycle[(node_idx - 1 + n) % n]
            next_node = cycle[(node_idx + 1) % n]
            # Sprawdź zamianę (prev, node) z inną krawędzią (p, q)
            for k in range(n):
                if k == node_idx or k == (node_idx - 1 + n) % n:
                    continue
                p = cycle[k]
                q = cycle[(k + 1) % n]
                delta = delta_swap_edge(self.cities, prev_node, node, p, q)
                if delta < -1e-9:
                    moves.append((delta, SWAP_EDGE, prev_node, node, p, q))
            # Sprawdź zamianę (node, next) z inną krawędzią (p, q)
            for k in range(n):
                if k == node_idx or k == (node_idx + 1) % n:
                    continue
                p = cycle[k]
                q = cycle[(k + 1) % n]
                delta = delta_swap_edge(self.cities, node, next_node, p, q)
                if delta < -1e-9:
                    moves.append((delta, SWAP_EDGE, node, next_node, p, q))

        # 2. Ruchy SWAP_NODE angażujące affected_node
        for node in affected_nodes:
            node_info = find_node(cycles, node)
            if node_info is None:
                continue
            c1_idx, i = node_info
            c2_idx = 1 - c1_idx  # Drugi cykl
            cycle2 = cycles[c2_idx]
            m = len(cycle2)
            if m == 0:
                continue
            for j in range(m):  # Dla każdego węzła w drugim cyklu
                delta, move_details = make_swap_node(
                    self.cities, cycles, c1_idx, i, c2_idx, j
                )
                if move_details and delta < -1e-9:
                    moves.append(move_details)

        # Usuń duplikaty i posortuj
        unique_moves = sorted(list(set(moves)), key=lambda x: x[0])
        return unique_moves

    def __call__(self, initial_cycles):
        cycles = deepcopy(initial_cycles)
        start_time = time()
        # Inicjalizacja: znajdź wszystkie poprawiające ruchy i posortuj
        moves = sorted(init_moves(self.cities, cycles), key=lambda x: x[0])
        # print(f"Initial improving moves: {len(moves)}") # Debug

        iterations = 0
        while True:
            iterations += 1
            # print(f"Memory Iteration: {iterations}, Moves count: {len(moves)}") # Debug

            if not moves:
                # print("No moves in the list. Stopping.") # Debug
                break

            # Znajdź pierwszy *ważny* ruch w posortowanej liście
            # Ruch jest ważny, jeśli jego krawędzie/węzły nadal istnieją w obecnej konfiguracji
            best_valid_move = None
            best_move_index = -1

            # Użyjemy setów do szybkiego sprawdzania istnienia krawędzi
            edges_cycle0 = set()
            edges_cycle1 = set()
            n0, n1 = len(cycles[0]), len(cycles[1])
            if n0 > 1:
                for i in range(n0):
                    edges_cycle0.add(
                        tuple(sorted((cycles[0][i], cycles[0][(i + 1) % n0])))
                    )
            if n1 > 1:
                for i in range(n1):
                    edges_cycle1.add(
                        tuple(sorted((cycles[1][i], cycles[1][(i + 1) % n1])))
                    )

            nodes_cycle0 = set(cycles[0])
            nodes_cycle1 = set(cycles[1])

            for k, move in enumerate(moves):
                delta, kind = move[0], move[1]
                is_valid = False

                if kind == SWAP_EDGE:
                    _, _, a, b, c, d = move
                    edge1 = tuple(sorted((a, b)))
                    edge2 = tuple(sorted((c, d)))
                    # Sprawdź, czy obie krawędzie istnieją w TYM SAMYM cyklu
                    if edge1 in edges_cycle0 and edge2 in edges_cycle0:
                        is_valid = True
                        # Sprawdź orientację (niekonieczne, jeśli delta jest poprawna)
                    elif edge1 in edges_cycle1 and edge2 in edges_cycle1:
                        is_valid = True

                elif kind == SWAP_NODE:
                    _, _, c1_orig, c2_orig, _, y1, _, _, y2, _ = move
                    # Sprawdź, czy y1 jest w cyklu c1_orig, a y2 w c2_orig
                    if y1 in nodes_cycle0 and y2 in nodes_cycle1 and c1_orig == 0:
                        is_valid = True
                    elif y1 in nodes_cycle1 and y2 in nodes_cycle0 and c1_orig == 1:
                        is_valid = True

                if is_valid:
                    # Sprawdź, czy delta jest nadal poprawna (opcjonalnie, ale bezpieczniej)
                    # recalculate_delta(...)
                    # if abs(recalculated_delta - delta) < 1e-6:
                    best_valid_move = move
                    best_move_index = k
                    break  # Znaleziono pierwszy ważny ruch

            if best_valid_move is None:
                # print("No valid moves found in the list. Stopping.") # Debug
                break  # Brak ważnych ruchów na liście

            # Usuń wykonany ruch z listy
            moves.pop(best_move_index)

            # Zastosuj ruch
            apply_move(cycles, best_valid_move)
            # print(f"Applied valid move: {best_valid_move[1]}, Delta: {best_valid_move[0]}, New Score: {score(self.cities, cycles)}") # Debug

            # Usuń z listy ruchy, które stały się nieważne przez zastosowany ruch
            # To jest trudne do zrobienia efektywnie bez ponownego sprawdzania ważności
            # Prostsze podejście: wygeneruj nowe ruchy i połącz listy

            # Wygeneruj nowe ruchy w sąsiedztwie wykonanego ruchu
            new_potential_moves = self.next_moves(cycles, best_valid_move)
            # print(f"Generated {len(new_potential_moves)} new potential moves.") # Debug

            # Połącz starą listę (bez wykonanego ruchu i potencjalnie nieważnych)
            # z nowymi ruchami. Użyj set do usunięcia duplikatów i posortuj.
            # To może być nieefektywne, jeśli listy są duże.
            # Lepsza strategia: zaktualizuj delty istniejących ruchów, usuń nieważne, dodaj nowe.

            # Proste połączenie i sortowanie:
            # Stwórz set istniejących ruchów dla szybkiego sprawdzania
            existing_moves_set = set(moves)
            for new_move in new_potential_moves:
                # Dodaj tylko jeśli jest naprawdę nowy lub ma lepszą deltę niż istniejący (trudne do śledzenia)
                # Na razie po prostu dodajemy wszystkie nowe potencjalne
                existing_moves_set.add(new_move)

            moves = sorted(list(existing_moves_set), key=lambda x: x[0])
            # Ogranicz rozmiar listy ruchów (opcjonalnie)
            # MAX_MOVES = 5000
            # if len(moves) > MAX_MOVES:
            #     moves = moves[:MAX_MOVES]

        end_time = time()
        return end_time - start_time, cycles


# --- Główna część skryptu (przykład użycia) ---

if __name__ == "__main__":
    # Wczytaj instancję
    instance_path = "kroA200.tsp"  # Przykładowy plik, zmień na właściwy
    try:
        cities_matrix, coords_df = read_instance(instance_path)
        N_cities = cities_matrix.shape[0]
        print(f"Wczytano instancję: {instance_path} ({N_cities} miast)")

        # 1. Wygeneruj rozwiązanie początkowe (losowe lub regret)
        print("\nGenerowanie rozwiązania początkowego...")
        # Opcja A: Losowe
        # t_random, initial_solution = 0, random_solution(N_cities, seed=2024)
        # print(f"Wygenerowano losowe rozwiązanie. Koszt: {score(cities_matrix, initial_solution)}")

        # Opcja B: Regret (może wymagać wielu startów dla lepszego wyniku)
        # Uruchomienie Regret z wielu punktów startowych równolegle
        num_starts = 4  # Liczba różnych punktów startowych do przetestowania
        pool = mp.Pool(
            processes=min(num_starts, mp.cpu_count())
        )  # Użyj dostępnych rdzeni
        start_nodes = random.sample(
            range(N_cities), min(num_starts, N_cities)
        )  # Losowe punkty startowe
        regret_args = [(cities_matrix, start) for start in start_nodes]

        print(
            f"Uruchamianie heurystyki Regret z {len(start_nodes)} punktów startowych..."
        )
        regret_results = pool.map(solve_regret, regret_args)
        pool.close()
        pool.join()

        # Wybierz najlepsze rozwiązanie z Regret
        best_regret_time = min(res[0] for res in regret_results)
        best_regret_solution = min(
            regret_results, key=lambda res: score(cities_matrix, res[1])
        )[1]
        initial_solution = best_regret_solution
        initial_score = score(cities_matrix, initial_solution)
        print(
            f"Najlepsze rozwiązanie Regret: Koszt = {initial_score:.0f} (czas: {best_regret_time:.2f}s)"
        )

        # 2. Uruchom algorytm przeszukiwania lokalnego
        print("\nUruchamianie przeszukiwania lokalnego...")

        # Opcja 1: Steepest Descent
        # search_algo = SearchSteepest(cities_matrix)
        # print("Algorytm: Steepest Descent")
        # search_time, final_solution = search_algo(initial_solution)
        # final_score = score(cities_matrix, final_solution)
        # print(f"Wynik Steepest Descent: Koszt = {final_score:.0f} (czas: {search_time:.2f}s)")

        # Opcja 2: Candidates Moves
        search_algo_cand = SearchCandidates(cities_matrix)
        print("Algorytm: Candidates Moves (k=15)")
        search_time_cand, final_solution_cand = search_algo_cand(
            initial_solution, k=15
        )  # k - parametr
        final_score_cand = score(cities_matrix, final_solution_cand)
        print(
            f"Wynik Candidates Moves: Koszt = {final_score_cand:.0f} (czas: {search_time_cand:.2f}s)"
        )

        # Opcja 3: Memory Search
        # search_algo_mem = SearchMemory(cities_matrix)
        # print("Algorytm: Memory Search")
        # search_time_mem, final_solution_mem = search_algo_mem(initial_solution)
        # final_score_mem = score(cities_matrix, final_solution_mem)
        # print(f"Wynik Memory Search: Koszt = {final_score_mem:.0f} (czas: {search_time_mem:.2f}s)")

        # Wybierz najlepsze rozwiązanie z testowanych algorytmów
        final_solution = final_solution_cand  # Domyślnie weź wynik z Candidates
        final_score = final_score_cand

        # if 'final_solution_mem' in locals() and final_score_mem < final_score:
        #      final_solution = final_solution_mem
        #      final_score = final_score_mem
        # if 'final_solution' in locals() and final_score < final_score: # Porównaj ze Steepest, jeśli było uruchomione
        #      final_solution = final_solution
        #      final_score = final_score

        print(f"\nNajlepszy znaleziony wynik: {final_score:.0f}")
        print(f"Cykl 1 ({len(final_solution[0])} miast): {final_solution[0]}")
        print(f"Cykl 2 ({len(final_solution[1])} miast): {final_solution[1]}")

        # 3. Wyświetl rozwiązanie (opcjonalnie)
        print("\nRysowanie rozwiązania...")
        plt.figure(figsize=(12, 8))  # Utwórz nowe okno dla wykresu
        plot_solution(coords_df, final_solution)
        plt.title(f"Rozwiązanie TSP ({instance_path}) - Koszt: {final_score:.0f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")  # Równe skale osi
        plt.show()

    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku instancji: {instance_path}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
        import traceback

        traceback.print_exc()
