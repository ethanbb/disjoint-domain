from typing import List, Literal, Callable
import numpy as np


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


class RelationshipFn:
    def __init__(self, rel_name: str, impl: Callable[[], List['FamilyMember']]):
        self.name = rel_name  # name of the relationship
        self.impl = impl
    
    def __call__(self):
        return self.impl()


class FamilyMember:
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
        return list(set(my_side + spouses_side))

    def get_piblings(self):
        """pibliings = aunts and uncles"""
        return self.parents.wife.get_siblings_w_inlaws() if self.parents is not None else []

    def get_uncles(self):
        return [pib for pib in self.get_piblings() if pib.sex == 'm']

    def get_aunts(self):
        return [pib for pib in self.get_piblings() if pib.sex == 'f']

    def get_niblings(self):
        """niblings = nieces and nephews"""
        return list(set(sum([sib.get_children() for sib in self.get_siblings_w_inlaws()], [])))

    def get_nephews(self):
        return [nib for nib in self.get_niblings() if nib.sex == 'm']

    def get_nieces(self):
        return [nib for nib in self.get_niblings() if nib.sex == 'f']
    
    def __init__(self, name: str, sex: Literal['m', 'f'], parents: Couple = None):
        self.name = name
        self.sex = sex
        self.parents = parents
        self.couple = None
        if parents is not None:
            parents.children.append(self)
    
        # Set of functions to compute relationships, in the order they are encoded in relationship input units
        self.relationship_fns = [
            RelationshipFn('father', self.get_fathers),
            RelationshipFn('mother', self.get_mothers),
            RelationshipFn('husband', self.get_husbands),
            RelationshipFn('wife', self.get_wives),
            RelationshipFn('sons', self.get_sons),
            RelationshipFn('daughters', self.get_daughters),
            RelationshipFn('brothers', self.get_brothers),
            RelationshipFn('sisters', self.get_sisters),
            RelationshipFn('uncles', self.get_uncles),
            RelationshipFn('aunts', self.get_aunts),
            RelationshipFn('nephews', self.get_nephews),
            RelationshipFn('nieces', self.get_nieces)
        ]


class FamilyTree:
    def __init__(self, members: List[FamilyMember], couples: List[Couple], validate=True):
        self.members = members
        self.couples = couples
        self.size = len(members)
        self.member_index = {member: i for (i, member) in enumerate(members)}

        if validate:
            for member in members:
                if member.parents is not None:
                    assert member.parents in couples, f'Tree invalid - parents of {member.name} not found'

            for couple in couples:
                for child in couple.children:
                    assert child in members, f'Tree invalid - child {child.name} not in members list'
        
    def __repr__(self):
        return f'FamilyTree({self.members!r}, {self.couples!r})'
    
    def __add__(self, other):
        if not isinstance(other, FamilyTree):
            return NotImplemented
        newtree = FamilyTree(self.members + other.members, self.couples + other.couples)
        return newtree

    def get_nonempty_related_members_mat(self, subject: FamilyMember, zeros_fn=np.zeros):
        """
        Get a matrix of which other family members are related to the subject by each relationship
        for which there is at least one related member, along with the corresponding matrix of relationship indices
        """
        related_members = [fn() for fn in subject.relationship_fns]
        rel_member_pairs = [(rel_ind, membs) for rel_ind, membs in enumerate(related_members) if len(membs) > 0]
        n_rels = len(rel_member_pairs)
        
        rel_mat = zeros_fn((n_rels, len(subject.relationship_fns)))
        member_mat = zeros_fn((n_rels, self.size))
        for i, (rel_ind, membs) in enumerate(rel_member_pairs):
            rel_mat[i, rel_ind] = 1
            for member in membs:
                member_mat[i, self.member_index[member]] = 1
        
        return rel_mat, member_mat
    
    def get_io_mats(self, zeros_fn=np.zeros, cat_fn=np.concatenate):
        """Get matrices encoding person1, relationship, and person2 for whole tree"""
        person1_mats = []
        rel_mats = []
        person2_mats = []
        
        for i, subject in enumerate(self.members):
            rel_mat, member_mat = self.get_nonempty_related_members_mat(subject, zeros_fn=zeros_fn)
            subj_mat = zeros_fn((rel_mat.shape[0], self.size))
            subj_mat[:, i] = 1
            person1_mats.append(subj_mat)
            rel_mats.append(rel_mat)
            person2_mats.append(member_mat)
        
        person1_mat = cat_fn(person1_mats, 0)
        rel_mat = cat_fn(rel_mats, 0)
        person2_mat = cat_fn(person2_mats, 0)
        
        return person1_mat, rel_mat, person2_mat
    
    def get_person2_mats_by_rel(self, zeros_fn=np.zeros, nan_if_none=False):
        """Get a list of person 2 matrices, one for each relationship, with 1 row per person 1"""
        n_rels = len(self.members[0].relationship_fns)
        mats = [zeros_fn((self.size, self.size)) for _ in range(n_rels)]
        
        for rel_ind in range(n_rels):
            for subj_ind, subject in enumerate(self.members):
                relatives = subject.relationship_fns[rel_ind]()
                if len(relatives) == 0 and nan_if_none:
                    mats[rel_ind][subj_ind, :] = np.nan
                else:
                    for relative in relatives:
                        mats[rel_ind][subj_ind, self.member_index[relative]] = 1   
        return mats


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


def get_french_tree():
    """Make another family tree that is not isomorphic to the Hinton trees"""
    albert = FamilyMember('Albert', 'm')
    alice = FamilyMember('Alice', 'f')
    alal = Couple(alice, albert)
    
    cedric = FamilyMember('Cedric', 'm', parents=alal)
    celine = FamilyMember('Celine', 'f', parents=alal)
    edouard = FamilyMember('Edouard', 'm', parents=alal)
    sarah = FamilyMember('Sarah', 'f')
    sared = Couple(sarah, edouard)
    
    marie = FamilyMember('Marie', 'f', parents=sared)
    louise = FamilyMember('Louise', 'f', parents=sared)
    robert = FamilyMember('Robert', 'm', parents=sared)
    jean = FamilyMember('Jean', 'm')
    marjean = Couple(marie, jean)
    juliette = FamilyMember('Juliette', 'f')
    julirob = Couple(juliette, robert)
    rene = FamilyMember('Rene', 'm', parents=julirob)
    
    members = [albert, alice, cedric, celine, edouard, sarah,
               marie, louise, robert, jean, juliette, rene]
    couples = [alal, sared, marjean, julirob]
    
    return FamilyTree(members, couples)
    

def get_tree(name='english'):
    return {
        'english': lambda: get_hinton_tree(italian=False),
        'italian': lambda: get_hinton_tree(italian=True),
        'french': get_french_tree
    }[name]()
