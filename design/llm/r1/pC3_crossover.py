# pC3 crossover: tilted region from one same-band donor n' (any of 8 symmetries), upright blocks from another
# (or our own n); uprights of U that fit around T's tilted squares are kept, T's own uprights fill the rest, count fixed
# by least-contact deletion / max-clearance insertion. Pair and symmetry picked by gap(T) + count-repair penalty.
_BOARD_TXT = """
257:16.53553390593 258:16.56344800213 259:16.59001643926 260:16.63557052055 261:16.67877987534 262:16.69811501873 263:16.72437270914 264:16.77028852775
265:16.77817459305 266:16.82302875076 267:16.83881526995 268:16.87186813514 269:16.89240780246 270:16.92731421642 271:16.94894773371 272:16.96550117587
273:16.98392596265 290:17.53553390593 291:17.53553390593 292:17.58905394337 293:17.62306521206 294:17.65685424949 295:17.69804292939 296:17.70710678119
297:17.74041017231 298:17.77688038938 299:17.79363699181 300:17.82412338848 301:17.84666719284 302:17.87001385040 303:17.89967388936 304:17.91800547284
305:17.94575583878 306:17.96343371749 307:17.97923096042 308:17.99714447002 325:18.53553390593 326:18.53553390593 327:18.58378219015 328:18.59953546481
329:18.64919843663 330:18.68379533751 331:18.70710678119 332:18.74041017228 333:18.76307892480 334:18.78404234273 335:18.79481592019 336:18.82359786906
337:18.85372898137 338:18.87761534045 339:18.89830010076 340:18.92041008197 341:18.93643554410 342:18.96250923902 343:18.97516265318 344:18.99073403446
362:19.53553390593 363:19.53553390593 364:19.57142857143 365:19.59953546476 366:19.64580204702 367:19.67972381412 368:19.69238815543 369:19.70710678119
370:19.75118176126 371:19.77346535278 372:19.78900010781 373:19.80001993076 374:19.82547547479 375:19.85553806992 376:19.87019507893 377:19.89407524475
378:19.91920597279 379:19.93829667994 380:19.95941164376 381:19.97376356273 382:19.98329404280 401:20.52614301403 402:20.53553390593 403:20.56493573028
404:20.59137814550 405:20.63016790066 406:20.65685424949 407:20.69094333415 408:20.70710678119 409:20.74041017228 410:20.76067720305 411:20.77413529708
412:20.79412933024 413:20.81521755481 414:20.84037472621 415:20.85659906849 416:20.87397821019 417:20.89510605365 418:20.91676757656 419:20.93274950508
420:20.95799511382 421:20.97176414881 422:20.98150274333 423:20.99371339104 442:21.52464793793 443:21.53553390593 444:21.56469316775 445:21.59001643928
446:21.62306521211 447:21.64836585350 448:21.68107773228 449:21.70158901830 450:21.72367731293 451:21.75066387738 452:21.76266192022 453:21.77798150091
454:21.79378784383 455:21.81645135567 456:21.83421758692 457:21.85970095517 458:21.87552157153 459:21.89499782980 460:21.91160004625 461:21.92827971060
462:21.94641637296 463:21.96798246650 464:21.97912709364 465:21.99031989890 485:22.52464793797 486:22.53553390593 487:22.56469316775 488:22.59001643928
489:22.60426191892 490:22.63619365368 491:22.67453091085 492:22.69054047576 493:22.70710678119 494:22.73454489201 495:22.75818641116 496:22.76442905015
497:22.78028724371 498:22.79386312376 499:22.82022847041 500:22.83837789096 501:22.85683989054 502:22.87411253394 503:22.89446497832 504:22.90971742348
505:22.92815479585 506:22.94193935093 507:22.96083370024 508:22.97701850592 509:22.98610845620 510:22.99710601757 530:23.52105896906 531:23.53553390593
532:23.56238461961 533:23.58378219015 534:23.59604311668 535:23.63577392581 536:23.65652496539 537:23.68414360645 538:23.70502452250 539:23.72443152123
540:23.74743639710 541:23.76153640180 542:23.77628803159 543:23.78726731960 544:23.79412954535 545:23.81276410951 546:23.83957665076 547:23.85770568579
548:23.87079324967 549:23.88950723471 550:23.90598277444 551:23.92040617886 552:23.93853276629 553:23.95794836726 554:23.97109285093 555:23.98381898206
556:23.99330904562 577:24.52068217232 578:24.53553390593 579:24.56233201090 580:24.58089876593 581:24.59377488184 582:24.62245850786 583:24.64633265957
584:24.67663629807 585:24.69741179702 586:24.70710678119 587:24.73561603084 588:24.75005608682 589:24.76381474199 590:24.77866035489 591:24.79107589301
592:24.79675994291 593:24.81540755432 594:24.84087201742 595:24.85519788283 596:24.86968251780 597:24.88996823639 598:24.90037462194 599:24.91668391340
600:24.93272835629 601:24.95358332332 602:24.96871078348 603:24.98084708827 604:24.99005119871 605:24.99995088874 626:25.51343464764 627:25.53553390593
628:25.53553390593 629:25.56600635499 630:25.59024971984 631:25.60871257855 632:25.63703692571 633:25.65685424949 634:25.68230889196 635:25.69873975406
636:25.72510141203 637:25.74141492121 638:25.75839149165 639:25.76990727930 640:25.77905365663 641:25.79389547841 642:25.81071681966 643:25.82946243948
644:25.84485441784 645:25.85564395036 646:25.86823032279 647:25.88627972686 648:25.89979011416 649:25.91029591730 650:25.92648747811 651:25.94098849814
652:25.96056261357 653:25.97847602979 654:25.98629009715 655:25.99580317558 677:26.51343464764 678:26.53054405645 679:26.53553390593 680:26.56562419471
681:26.59001643929 682:26.60658969879 683:26.63130658468 684:26.64002285000 685:26.67868979775 686:26.68672355332 687:26.70641401480 688:26.72792206136
689:26.74241775504 690:26.76098078163 691:26.77582777526 692:26.78024905978 693:26.79254438342 694:26.81417693298 695:26.82399157743 696:26.84488028546
697:26.85649653434 698:26.87227752189 699:26.88434591940 700:26.89722128570 701:26.91041719430 702:26.92034169428 703:26.93443294875 704:26.95793708188
705:26.97068497198 706:26.98422209896 707:26.99181021355 730:27.51343464771 731:27.53053971819 732:27.53553390593 733:27.56562419471 734:27.59001643929
735:27.59787683649 736:27.62669159966 737:27.63882856854 738:27.66549236745 739:27.68414360637 740:27.69740481245 741:27.70721317783 742:27.73109778886
743:27.75023776792 744:27.75748341627 745:27.77601519949 746:27.78182898564 747:27.79375965706 748:27.81286236268 749:27.82609110452 750:27.83738644416
751:27.85808092268 752:27.87245458375 753:27.88376525430 754:27.89854694717 755:27.90730647042 756:27.92102721453 757:27.93177095452 758:27.94766097541
759:27.96576284697 760:27.97620980377 761:27.98917468166 762:27.99658126773 785:28.51343464764 786:28.53045738045 787:28.53553390593 788:28.56316249946
789:28.58905394339 790:28.59323950409 791:28.60640037581 792:28.62959013877 793:28.65027742388 794:28.67381257517 795:28.68748538303 796:28.70540122151
797:28.72782263134 798:28.73622986390 799:28.75324232677 800:28.76497407037 801:28.78013288278 802:28.78791792674 803:28.79394344023 804:28.81006058541
805:28.82893226344 806:28.84075158980 807:28.85546533134 808:28.86875765852 809:28.88248241073 810:28.89371899936 811:28.90739895959 812:28.91853405224
813:28.92874461275 814:28.94424748925 815:28.95870726780 816:28.97425048067 817:28.98361474803 818:28.99400989189 842:29.50989197458 843:29.52745178050
844:29.53553390593 845:29.56308664664 846:29.58089876594 847:29.59323889933 848:29.60588382953 849:29.62737482379 850:29.63593974028 851:29.66264578260
852:29.68214534796 853:29.69165841196 854:29.70588235300 855:29.73162946479 856:29.74396798636 857:29.75580212857 858:29.76995565547 859:29.78150964443
860:29.79210647138 861:29.79541463077 862:29.81208645595 863:29.83033311292 864:29.84096753431 865:29.85405375248 866:29.86575714181 867:29.87990143540
868:29.89085044109 869:29.90409372298 870:29.91437752839 871:29.92784681703 872:29.93630643118 873:29.95620129862 874:29.96955107557 875:29.98085712167
876:29.99083340981 877:29.99885719010 901:30.50659102934 902:30.52427210294 903:30.53553390593 904:30.54975498143 905:30.56682249063 906:30.59137814550
907:30.60234822878 908:30.60588450859 909:30.63425024814 910:30.64688404554 911:30.66091616065 912:30.68749749889 913:30.69733985195 914:30.72218342559
915:30.73176074155 916:30.74614125328 917:30.75586967182 918:30.76804305517 919:30.78213276968 920:30.79251628456 921:30.80571877990 922:30.82135678232
923:30.83522682520 924:30.84161846897 925:30.85457921826 926:30.86384776852 927:30.88068055743 928:30.88932189193 929:30.90065217028 930:30.91269111373
931:30.92110319157 932:30.93549965775 933:30.94375771639 934:30.96205251039 935:30.97438011807 936:30.98526420222 937:30.99476750729 962:31.50659102934
963:31.52262363886 964:31.53403099414 965:31.54521155450 966:31.56239505557 967:31.59001643930 968:31.59991763195 969:31.60588450855 970:31.62947149590
971:31.63678931465 972:31.65140129865 973:31.67218808724 974:31.68837991002 975:31.70162895378 976:31.72363686259 977:31.73985060383 978:31.74889187396
979:31.76362273009 980:31.77053496778 981:31.78102706793 982:31.79310649630 983:31.81130879302 984:31.82040400138 985:31.83548822845 986:31.84492810203
987:31.85473223280 988:31.86436331559 989:31.87449703618 990:31.88866949674 991:31.89818393903 992:31.91129819276 993:31.91916745813 994:31.92834951233
995:31.94265058486 996:31.95332299202 997:31.97151913269 998:31.98281417243 999:31.99175502908 1000:31.99878478638
"""

# ---------------- shared transfer toolkit (self-contained) ----------------
import os, csv, math
import numpy as np
from scipy.spatial import cKDTree

_DB = None
_BD = None


def _db():
    """all our packings n -> (n,3) array, loaded once from ../base.csv."""
    global _DB
    if _DB is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'base.csv')
        rows = {}
        for r in csv.DictReader(open(p)):
            rows.setdefault(int(r['n']), []).append((int(r['square_id']), float(r['x']), float(r['y']), float(r['angle'])))
        _DB = {k: np.array([v[1:] for v in sorted(rs)]) for k, rs in rows.items()}
    return _DB


def _board(m):
    global _BD
    if _BD is None:
        _BD = {}
        for tok in _BOARD_TXT.split():
            k, v = tok.split(':'); _BD[int(k)] = float(v)
    return _BD.get(m, float(math.ceil(math.sqrt(m))))


def Kof(m):
    return int(math.ceil(math.sqrt(m)))


def cof(m):
    return m - (Kof(m) - 1) ** 2


def halfext(t):
    return 0.5 * (np.abs(np.cos(t)) + np.abs(np.sin(t)))


def side(a):
    h = halfext(a[:, 2])
    return max((a[:, 0] + h).max() - (a[:, 0] - h).min(), (a[:, 1] + h).max() - (a[:, 1] - h).min())


def normalize(a):
    a = a.copy(); h = halfext(a[:, 2])
    a[:, 0] -= (a[:, 0] - h).min(); a[:, 1] -= (a[:, 1] - h).min()
    return a


def gap(m):
    return side(_db()[m]) - _board(m)


def is_up(a, tol=1e-6):
    return np.abs(np.sin(2 * a[:, 2])) < tol


def sym(a, k, S):
    """dihedral symmetry k in 0..7 of a packing living in [0,S]^2."""
    a = a.copy(); x, y, t = a[:, 0].copy(), a[:, 1].copy(), a[:, 2].copy()
    if k & 1:
        x, y, t = y, x, np.pi / 2 - t
    if k & 2:
        x, t = S - x, -t
    if k & 4:
        y, t = S - y, -t
    a[:, 0], a[:, 1], a[:, 2] = x, y, np.mod(t, np.pi / 2)
    return a


def _corners(a):
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    off = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    X = a[:, None, 0] + off[None, :, 0] * c[:, None] - off[None, :, 1] * s[:, None]
    Y = a[:, None, 1] + off[None, :, 0] * s[:, None] + off[None, :, 1] * c[:, None]
    return np.stack([X, Y], -1)


def pair_sat(a, I, J):
    """for pairs (I,J): min-axis overlap (positive = penetration, negative = separation) and that axis."""
    P = _corners(a)
    best = np.full(len(I), 9.0); bax = np.zeros((len(I), 2))
    for src in (I, J):
        t = a[src, 2]
        for ax in (np.stack([np.cos(t), np.sin(t)], 1), np.stack([-np.sin(t), np.cos(t)], 1)):
            pi = np.einsum('kcd,kd->kc', P[I], ax); pj = np.einsum('kcd,kd->kc', P[J], ax)
            ov = np.minimum(pi.max(1), pj.max(1)) - np.maximum(pi.min(1), pj.min(1))
            m = ov < best; best[m] = ov[m]; bax[m] = ax[m]
    return best, bax


def pairs(a, r=1.4143):
    if len(a) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    pr = cKDTree(a[:, :2]).query_pairs(r, output_type='ndarray')
    if len(pr) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    return pr[:, 0], pr[:, 1]


def depth_sum(a):
    I, J = pairs(a)
    if len(I) == 0:
        return 0.0
    d, _ = pair_sat(a, I, J)
    return float(d[d > 1e-9].sum())


def contacts(a, S, eps=0.02):
    """number of touching neighbours (incl. walls) per square."""
    I, J = pairs(a); cnt = np.zeros(len(a))
    if len(I):
        d, _ = pair_sat(a, I, J); m = d > -eps
        np.add.at(cnt, I[m], 1); np.add.at(cnt, J[m], 1)
    h = halfext(a[:, 2])
    for v in (a[:, 0] - h, S - a[:, 0] - h, a[:, 1] - h, S - a[:, 1] - h):
        cnt += v < eps
    return cnt


def remove_least_contact(a, k, S):
    """delete k squares, each time the one with fewest contacts (ties: upright first, then interior)."""
    a = a.copy()
    for _ in range(k):
        c = contacts(a, S)
        ctr = np.hypot(a[:, 0] - S / 2, a[:, 1] - S / 2)
        key = c * 10 + (~is_up(a)) * 2 - ctr / S
        a = np.delete(a, int(np.argmin(key)), 0)
    return a


def cand_depth(a, P, theta, S, tree=None):
    """overlap (sum of SAT depths, plus wall violation) of a unit square at points P with angle theta."""
    tree = tree or cKDTree(a[:, :2])
    nb = tree.query_ball_point(P, 1.4143)
    I = np.repeat(np.arange(len(P)), [len(x) for x in nb]); J = np.concatenate([np.array(x, int) for x in nb] + [np.zeros(0, int)])
    tot = np.zeros(len(P)); h = halfext(np.array([theta]))[0]
    tot += np.maximum(0, h - P[:, 0]) + np.maximum(0, P[:, 0] + h - S) + np.maximum(0, h - P[:, 1]) + np.maximum(0, P[:, 1] + h - S)
    if len(I):
        b = np.vstack([a, np.column_stack([P, np.full(len(P), theta)])])
        d, _ = pair_sat(b, len(a) + I, J)
        np.add.at(tot, I, np.maximum(d, 0))
    return tot


def insert_max_clear(a, k, S, step=0.08):
    """insert k squares, each at the grid point / angle of least overlap (max clearance)."""
    a = a.copy()
    g = np.arange(0.5, S - 0.5 + 1e-9, step)
    P = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    for _ in range(k):
        tree = cKDTree(a[:, :2])
        # discard points too close to any centre
        dd, _ = tree.query(P)
        Q = P[dd > 0.55]
        if len(Q) == 0:
            Q = P
        best = (1e9, None, 0)
        angs = [0.0] + list(np.unique(np.round(np.mod(a[~is_up(a), 2], np.pi / 2), 3))[:4])
        for th in angs:
            v = cand_depth(a, Q, th, S, tree)
            i = int(np.argmin(v))
            if v[i] < best[0]:
                best = (v[i], Q[i], th)
        a = np.vstack([a, [best[1][0], best[1][1], best[2]]])
    return a


def fix_count(a, n, S):
    if len(a) > n:
        return remove_least_contact(a, len(a) - n, S)
    if len(a) < n:
        return insert_max_clear(a, n - len(a), S)
    return a


def push_relax(a, S, iters=300, rot=True):
    """cheap overlap removal at fixed box S: move pairs apart along the min-penetration axis, clamp to walls."""
    a = a.copy()
    I, J = pairs(a, 1.6)
    for it in range(iters):
        if it % 15 == 0:
            I, J = pairs(a, 1.6)
        if len(I):
            d, ax = pair_sat(a, I, J)
            m = d > 1e-7
            if not m.any() and it % 15 != 14:
                h = halfext(a[:, 2])
                if np.all((a[:, 0] >= h - 1e-9) & (a[:, 0] <= S - h + 1e-9) & (a[:, 1] >= h - 1e-9) & (a[:, 1] <= S - h + 1e-9)):
                    break
            i, j, d, ax = I[m], J[m], d[m], ax[m]
            sgn = np.sign(np.einsum('kd,kd->k', a[i, :2] - a[j, :2], ax)); sgn[sgn == 0] = 1
            mv = (0.5 * d + 1e-4)[:, None] * ax * sgn[:, None]
            D = np.zeros((len(a), 2)); np.add.at(D, i, mv); np.add.at(D, j, -mv)
            a[:, :2] += 0.7 * D
        h = halfext(a[:, 2])
        a[:, 0] = np.clip(a[:, 0], h, S - h); a[:, 1] = np.clip(a[:, 1], h, S - h)
    return a


def fit_box(a, S):
    """uniformly rescale a normalized packing (about the centre) so its side becomes S."""
    a = normalize(a); s0 = side(a)
    a[:, :2] = S / 2 + (a[:, :2] - s0 / 2) * (S / s0)
    return a


def finish(a, n, S, rng, jitter=0.0):
    a = fix_count(a, n, S)
    a = push_relax(a, S)
    if jitter > 0:
        a[:, :2] += rng.normal(0, jitter, (len(a), 2))
        a = push_relax(a, S, 60)
    a[:, 2] = np.mod(a[:, 2], np.pi / 2)
    return a
# ---------------- end toolkit ----------------


def _free(a, U, S):
    """mask of upright squares U that do not overlap packing a (depth < 1e-3)."""
    if len(U) == 0:
        return np.zeros(0, bool)
    if len(a) == 0:
        return np.ones(len(U), bool)
    return cand_depth(a, U[:, :2], 0.0, S) < 1e-3


def cross(T, U, S):
    """tilted squares of T + every upright of U that fits + every upright of T that still fits."""
    tT = T[~is_up(T)]; uU = U[is_up(U)].copy(); uT = T[is_up(T)].copy()
    uU[:, 2] = 0.0; uT[:, 2] = 0.0
    a = tT.copy()
    a = np.vstack([a, uU[_free(a, uU, S)]])
    a = np.vstack([a, uT[_free(a, uT, S)]])
    return a, int(_free(tT, uU, S).sum())


_PICK = {}


def pick(n, S):
    """T (tilted donor) and U (upright donor) from n'=n-4..n+4 in the same band (U may be n itself), 8 symmetries
    of T. score = gap(T) + 0.003 per square missing after the cross + 0.0005 per extra square; require that at
    least 10 uprights really come from U (otherwise it is just T)."""
    if n in _PICK:
        return _PICK[n]
    db = _db(); K = Kof(n)
    pool = sorted([(gap(m), m) for m in range(n - 4, n + 5) if m in db and Kof(m) == K])
    Ts = [m for _, m in pool[:4]]; Us = [m for _, m in pool[:5]]
    if n not in Us:
        Us.append(n)
    best = None
    for tm in Ts:
        T0 = fit_box(db[tm], S)
        for um in Us:
            if um == tm:
                continue
            U = fit_box(db[um], S)
            for k in range(8):
                T = sym(T0, k, S)
                a, fromU = cross(T, U, S)
                diff = len(a) - n
                sc = gap(tm) + 0.003 * max(0, -diff) + 0.0005 * max(0, diff)
                moved = _novel(T, U, S)
                if moved < 10:
                    continue
                if best is None or sc < best[0]:
                    best = (sc, tm, um, k, a)
    _PICK[n] = best
    return best


def _novel(T, U, S):
    """number of U uprights not coinciding with a T upright (centre distance > 0.1)."""
    uU = U[is_up(U)]; uT = T[is_up(T)]
    if len(uT) == 0 or len(uU) == 0:
        return len(uU)
    d, _ = cKDTree(uT[:, :2]).query(uU[:, :2])
    return int((d > 0.1).sum())


def build(n, base, board, rng):
    S = side(base)
    b = pick(n, S)
    if b is None:
        return base
    a = b[4].copy()
    a = finish(a, n, S, rng, jitter=0.002)
    return a
