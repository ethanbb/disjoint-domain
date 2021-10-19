from typing import List, Literal


class Couple:
    def __init__(self, wife: 'FamilyMember', husband: 'FamilyMember'):
        assert wife.couple is None and husband.couple is None, "Can't make couple from already-married members"
        self.wife = wife
        self.husband = husband
        wife.couple = self
        husband.couple = self
        self.children = []

    def __str__(self):
        return f'({self.wife}, {self.husband})'

    def __repr__(self):
        return f'Couple({self.wife!r}, {self.husband!r})'


class FamilyMember:
    def __init__(self, name: str, sex: Literal['m', 'f'], parents: Couple = None):
        self.name = name
        self.sex = sex
        self.parents = parents
        self.couple = None
        if parents is not None:
            parents.children.append(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'FamilyMember({self.name!r}, sex={self.sex!r}, parents={self.parents!r})'

    def get_children(self):
        return self.couple.children if self.couple is not None else []

    def get_siblings(self):
        return [c for c in self.parents.children if c is not self] if self.parents is not None else []

    # Functions to get all relations of each type
    def get_fathers(self):
        return [self.parents.husband] if self.parents is not None else []

    def get_mothers(self):
        return [self.parents.wife] if self.parents is not None else []

    def get_husbands(self):
        return [self.couple.husband] if self.couple is not None and self.sex == 'f' else []

    def get_wives(self):
        return [self.couple.wife] if self.couple is not None and self.sex == 'm' else []

    def get_sons(self):
        return [child for child in self.get_children() if child.sex == 'm']

    def get_daughters(self):
        return [child for child in self.get_children() if child.sex == 'f']

    def get_brothers(self):
        return [sib for sib in self.get_siblings() if sib.sex == 'm']

    def get_sisters(self):
        return [sib for sib in self.get_siblings() if sib.sex == 'f']

    def get_spouses(self):
        return self.get_husbands() + self.get_wives()

    def get_siblings_w_inlaws(self):
        def get_own_siblings_with_inlaws(member):
            return sum([[sib] + sib.get_spouses() for sib in member.get_siblings()], [])

        my_side = get_own_siblings_with_inlaws(self)
        spouses = self.get_spouses()
        spouses_side = get_own_siblings_with_inlaws(spouses[0]) if len(spouses) > 0 else []
        return my_side + spouses_side

    def get_piblings(self):
        """pibliings = aunts and uncles"""
        return self.parents.wife.get_siblings_w_inlaws() if self.parents is not None else []

    def get_uncles(self):
        return [pib for pib in self.get_piblings() if pib.sex == 'm']

    def get_aunts(self):
        return [pib for pib in self.get_piblings() if pib.sex == 'f']

    def get_niblings(self):
        """niblings = nieces and nephews"""
        return sum([sib.get_children() for sib in self.get_siblings_w_inlaws()], [])

    def get_nephews(self):
        return [nib for nib in self.get_niblings() if nib.sex == 'm']

    def get_nieces(self):
        return [nib for nib in self.get_niblings() if nib.sex == 'f']


class FamilyTree:
    def __init__(self, members: List[FamilyMember], couples: List[Couple], validate=True):
        self.members = members
        self.couples = couples

        if validate:
            for member in members:
                if member.parents is not None:
                    assert member.parents in couples, f'Tree invalid - parents of {member.name} not found'

            for couple in couples:
                for child in couple.children:
                    assert child in members, f'Tree invalid - child {child.name} not in members list'


def get_hinton_tree(italian=False):
    """Make the family tree defined in the Hinton paper"""
    christopher = FamilyMember('Roberto' if italian else 'Christopher', 'm')
    penelope = FamilyMember('Maria' if italian else 'Penelope', 'f')
    chrispen = Couple(penelope, christopher)

    andrew = FamilyMember('Pierro' if italian else 'Andrew', 'm')
    christine = FamilyMember('Francesca' if italian else 'Christine', 'f')
    andchris = Couple(christine, andrew)

    margaret = FamilyMember('Gina' if italian else 'Margaret', 'f')
    arthur = FamilyMember('Emilio' if italian else 'Arthur', 'm', parents=chrispen)
    margart = Couple(margaret, arthur)

    victoria = FamilyMember('Lucia' if italian else 'Victoria', 'f', parents=chrispen)
    james = FamilyMember('Marco' if italian else 'James', 'm', parents=andchris)
    vicjames = Couple(victoria, james)

    jennifer = FamilyMember('Angela' if italian else 'Jennifer', 'f', parents=andchris)
    charles = FamilyMember('Tomaso' if italian else 'Charles', 'm')
    jenncharles = Couple(jennifer, charles)

    colin = FamilyMember('Alfonso' if italian else 'Colin', 'm', parents=vicjames)
    charlotte = FamilyMember('Sophia' if italian else 'Charlotte', 'f', parents=vicjames)

    members = [christopher, penelope, andrew, christine, margaret, arthur,
               victoria, james, jennifer, charles, colin, charlotte]
    couples = [chrispen, andchris, margart, vicjames, jenncharles]

    return FamilyTree(members, couples)
