"""Example data sets.

   For now, they are used essencially for testing purposes."""
from abc import ABCMeta, abstractmethod
import pandas as pd
import six

class DataSetBase(metaclass=ABCMeta):
    """ Base class for an example data set"""

    def __init__(self, **kwargs):
        """ Constructor """
        pass

    @abstractmethod
    def as_pandas(self):
        """ Abstract method to return the dataset as pandas DataFrame."""
        pass

    @abstractmethod
    def as_str(self):
        """ Abstract method to return the dataset as a string."""
        pass

    @abstractmethod
    def create_example_file(self, file_name):
        """ Abstract method to create a file with the dataset of appropriate type."""
        pass

class DataSetFactory:
    """ The factory class for creating example data sets"""

    registry = {}
    """ Internal registry for available datasets"""

    @classmethod
    def register(cls, name):
        """ Class method to register a data set to the internal registry.
        Args:
            name (str): The name of the data set.
        Returns:
            The data set class itself.
        """

        def inner_wrapper(wrapped_class):
            # if already exists: replaces entry
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name, **kwargs):
        """Factory command to create the data set.
        This method gets the appropriate DataSet class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.
        Args:
            name (str): The name of the data set to create.
        Returns:
            An instance of the data set that is created.
        """

        if name not in cls.registry:
            raise AttributeError('Data set {} is not registered'.format(name))


        dataset_class = cls.registry[name]
        dataset = dataset_class(**kwargs)
        return dataset


@DataSetFactory.register('demo1')
class Demo1(DataSetBase):

    def as_str(self):
        return """m/z	s38	s39	s40	s32	s33	s34
97.58868	1073218	1049440	1058971	2351567	1909877	2197036
97.59001	637830	534900	582966	1440216	1124346	1421899
97.59185	460092	486631		1137139	926038	1176756
97.72992			506345			439583
98.34894	2232032	2165052	1966283			
98.35078	3255288	2813578	2516386			
98.35122		2499163	2164976			
98.36001			1270764	1463557	1390574	
98.57354				4627491	6142759	
98.57382		3721991	3338506			4208438
98.57497	6229543	3347404	2327096			
98.57528				2510001	1989197	4377331
98.57599	6897403	3946118				
98.57621			3242232	2467520		4314818
98.57692	8116811	5708658	3899578			
98.57712				2418202	986128	4946201
98.57790	3891025	3990442	3888258	2133404		3643682
98.57899		1877649	1864650	1573559		1829208
99.28772	2038979				3476845	"""

    def as_pandas(self):
        return pd.read_csv(six.StringIO(self.as_str()), sep='\t', index_col=0)

    def create_example_file(self, file_name):
        df = self.as_pandas()
        df.to_csv(file_name)

@DataSetFactory.register('demo2')
class Demo2(DataSetBase):

    def as_str(self):
        return """label	l1	l2	l1	l2	l2	l2
sample	s38	s39	s40	s32	s33	s34
m/z
97.58868	1073218	1049440	1058971	2351567	1909877	2197036
97.59001	637830	534900	582966	1440216	1124346	1421899
97.59185	460092	486631		1137139	926038	1176756
97.72992			506345			439583
98.34894	2232032	2165052	1966283			
98.35078	3255288		2516386			
98.35122		2499163	2164976			
98.36001			1270764	1463557	1390574	
98.57354				4627491	6142759	
98.57382		3721991	3338506			4208438
98.57497	6229543	3347404	2327096			
98.57528				2510001	1989197	4377331
98.57599	6897403	3946118				
98.57621			3242232	2467520		4314818
98.57692	8116811	5708658	3899578			
98.57712				2418202	986128	4946201
98.57790	3891025	3990442	3888258	2133404		3643682
98.57899		1877649	1864650	1573559		1829208
99.28772	2038979				3476845	"""


    def as_pandas(self):
        return pd.read_csv(six.StringIO(self.as_str()), sep='\t', index_col=0, header=[0,1])

    def create_example_file(self, file_name):
        df = self.as_pandas()
        df.to_csv(file_name)


@DataSetFactory.register('table_with_formulae')
class TableWithFormulae(DataSetBase):

    def as_str(self):
        return """Name	formula	Neutral mass
(+)-(1R,2R)-1,2-Diphenylethane-1,2-diol ([M+H]+)	C14H14O2	215.1066562
(+)-(1R,2R)-1,2-Diphenylethane-1,2-diol (see KEGG C16015) [secondary alcohol] ([M+H]+)	C14H14O2	215.1066562
(+)-10-methyl lauric acid; 10-methyl-dodecanoic acid [Branched fatty acids [FA0102]] ([M+H]+)	C13H26O2	215.2005565
(+)-10-methyl lauric acid; 10-methyl-dodecanoic acid [Branched fatty acids [FA0102]] ([M+Na]+)		237.1825008
(+)-12-(2-Cyclopenten-1-yl)-2-dodecanone ([M+H]+)	C17H30O	251.236942
(+)-12-hydroxy-9Z-hexadecenoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)	C16H30O3	271.2267713
(+)-12-hydroxy-9Z-hexadecenoic acid [Hydroxy fatty acids [FA0105]] ([M+Na]+)	C16H30O3	293.2087155
(+)-12-methyl myristic acid; 12-methyl-tetradecanoic acid [Branched fatty acids [FA0102]] ([M+H]+)	C15H30O2	243.2318567
(+)-12-methyl myristic acid; 12-methyl-tetradecanoic acid [Branched fatty acids [FA0102]] ([M+Na]+)	C15H30O2	265.2138009
(+)-13(17),15-Cleistanthadiene [C20 isoprenoids (diterpenes) [PR0104]] ([M+H]+)	C20H32	273.2576775
(+)-14,15-Epoxy-thapsan-14-ol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H26O2	239.2005565
(+)-14,15-Epoxy-thapsan-14-ol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+K39]+)	C15H26O2	277.1564395
(+)-14-methyl palmitic acid; 14-methyl-hexadecanoic acid [Branched fatty acids [FA0102]] ([M+H]+)	C17H34O2	271.2631568
(+)-14-methyl palmitic acid; 14-methyl-hexadecanoic acid [Branched fatty acids [FA0102]] ([M+Na]+)	C17H34O2	293.245101
(+)-15-Beyerene [C20 isoprenoids (diterpenes) [PR0104]] ([M+H]+)	C20H32	273.2576775
(+)-15S-hydroxy-hexadecanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)	C16H32O3	273.2424214
(+)-15S-hydroxy-hexadecanoic acid [Hydroxy fatty acids [FA0105]] ([M+Na]+)	C16H32O3	295.2243656
(+)-16-methyl stearic acid; 16-methyl-octadecanoic acid [Branched fatty acids [FA0102]] ([M+H]+)	C19H38O2	299.2944569
(+)-18-Hydroxy-7,16-sacculatadiene-11,12-dial [C20 isoprenoids (diterpenes) [PR0104]] ([M+Na]+)	C20H30O3	341.2087155
(+)-2-Himachalen-7-ol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H26O	223.2056419
(+)-2-Sterpurene-6,12,15-triol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H24O3	253.1798211
(+)-2-Sterpurene-6-ol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H24O	221.1899919
(+)-7-Isojasmonic acid ([M+H]+)	C12H18O3	211.1328709
(+)-7-Isomethyljasmonate ([M+H]+)	C13H20O3	225.148521
(+)-7beta-Hydroxy-15-beyeren-19-oic acid [C20 isoprenoids (diterpenes) [PR0104]] ([M+Na]+)	C20H30O3	341.2087155
(+)-8-Daucen-5-ol [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H26O	223.2056419
(+)-8-Drimen-7-one [C15 isoprenoids (sesquiterpenes) [PR0103]] ([M+H]+)	C15H24O	221.1899919
(+)-9,10,18-trihydroxy-12Z-octadecenoic acid [Hydroxy fatty acids [FA0105]] ([M+K39]+)	C18H34O5	369.2037836
(+)-Acutifolin A [Flavans, Flavanols and Leucoanthocyanidins [PK1202]] ([M+H]+)	C20H22O4	327.1590857"""

    def as_pandas(self):
        return pd.read_csv(six.StringIO(self.as_str()), sep='\t', index_col=None).set_index('Name')

    def create_example_file(self, file_name):
        df = self.as_pandas()
        df.to_csv(file_name)

@DataSetFactory.register('masstrix_output')
class MasstrixOutput(DataSetBase):

    def as_str(self):
        return """133.08577	0.00E+00	132.0784935	26	133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401#133.085920710401	-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#-1.1324306198618#	C03264#C03499#C06103#C07834#HMDB00317#HMDB00409#HMDB00525#HMDB00624#HMDB00665#HMDB00746#HMDB01624#HMDB01975#HMDB10718#HMDB12843#LMFA01050011#LMFA01050012#LMFA01050013#LMFA01050014#LMFA01050015#LMFA01050376#LMFA01050377#LMFA01050380#LMFA01050381#LMFA01050387#LMFA01050402#LMFA01050408	C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3#C6H12O3	"D-2-Hydroxyisocaproate ([M+H]+)#Ethyl (R)-3-hydroxybutanoate ([M+H]+)#6-Hydroxyhexanoic acid;6-Hydroxyhexanoate ([M+H]+)#Paraldehyde ([M+H]+)#2-Hydroxy-3-methylpentanoic acid; (2R,3R)-2-hydroxy-3-methyl-pentanoic acid [secondary alcohol] ([M+H]+)#(5R)-5-Hydroxyhexanoic acid; (5R)-5-hydroxyhexanoic acid [secondary alcohol] ([M+H]+)#5-Hydroxyhexanoic acid; 5-hydroxyhexanoic acid [secondary alcohol] ([M+H]+)#D-Leucic acid (see KEGG C03264); (2R)-2-hydroxy-4-methylpentanoic acid [secondary alcohol] ([M+H]+)#Leucinic acid; 2-hydroxy-4-methylpentanoic acid [secondary alcohol] ([M+H]+)#Hydroxyisocaproic acid; (2S)-2-hydroxy-4-methylpentanoic acid [secondary alcohol] ([M+H]+)#2-Hydroxycaproic acid; 2-hydroxyhexanoic acid [secondary alcohol] ([M+H]+)#2-Ethyl-2-Hydroxybutyric acid; 2-ethyl-2-hydroxy-butanoic acid [tertiary alcohol] ([M+H]+)#(R)-3-Hydroxyhexanoic acid [secondary alcohol] ([M+H]+)#6-Hydroxyhexanoic acid (see KEGG C06103); 6-hydroxyhexanoate [primary alcohol] ([M+H]+)#DL-2-hydroxy caproic acid; 2-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#DL-3-hydroxy caproic acid; 3-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#DL-4-hydroxy caproic acid; 4-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#5-hydroxy caproic acid; 5-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#6-hydroxy caproic acid; 6-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#5R-hydroxy-hexanoic acid; 5R-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#3R-hydroxy-hexanoic acid; 3R-hydroxy-hexanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#2-hydroxy-3-methyl-pentanoic acid; 2R-hydroxy-3R-methyl-pentanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#Leucinic acid; 2-hydroxy-4-methyl-pentanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#2-ethyl-2-hydroxy-butyric acid; 2-ethyl-2-hydroxy-butanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#D-Leucic acid; 2R-hydroxy-4-methyl-pentanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)#hydroxy-isocaproic acid; 2S-hydroxy-4-methyl-pentanoic acid [Hydroxy fatty acids [FA0105]] ([M+H]+)"													null#null#ko00930#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null	"null#null#;Caprolactam degradation#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null"	null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null#null
149.01145	0.00E+00	110.0482906	3	149.011172241631#149.011172241631#149.011172241631	1.86400688677955#1.86400688677955#1.86400688677955	C02560#C05130#HMDB03905	C5H6N2O#C5H6N2O#C5H6N2O	"N-Acetylimidazole ([M+K39]+)#Imidazole-4-acetaldehyde;Imidazole acetaldehyde ([M+K39]+)#Imidazole-4-acetaldehyde (see KEGG C05130); 2-(3H-imidazol-4-yl)acetaldehyde [aldehyde] ([M+K39]+)"													null#ko00340#null	"null#;Histidine metabolism#null"	null#null#null
177.00634	0.00E+00	138.0431806	4	177.006086861191#177.006086861191#177.006086861191#177.006086861191	1.43011153725101#1.43011153725101#1.43011153725101#1.43011153725101	C00785#C02126#HMDB00301#HMDB02730	C6H6N2O2#C6H6N2O2#C6H6N2O2#C6H6N2O2	"Urocanate;Urocanic acid ([M+K39]+)#4-Nitroaniline;p-Nitroaniline;4-Nitrobenzeneamine ([M+K39]+)#Urocanic acid (see KEGG C00785); 3-(3H-imidazol-4-yl)prop-2-enoic acid [carboxylic acid] ([M+K39]+)#Nicotinamide N-oxide; 1-oxidopyridin-1-ium-3-carboxamide [N-oxide] ([M+K39]+)"													"ko00340;ko01100#null#null#null"	";Histidine metabolism;Metabolic pathways#null#null#null"	null#null#null#null
189.04265	0.00E+00	150.0794906	1	189.0424724	0.939629703	C17512	C8H10N2O	"N-Methylanthranilamide;N-Methyl-2-aminobenzamide ([M+K39]+)"													null	null	null
203.05828	0.00E+00	164.0951206	3	203.057968967081#203.058122434051#203.058122434051	1.53174211364742#0.775964166523288#0.775964166523288	C00604#C11224#HMDB12246	C12H8N2#C9H12N2O#C9H12N2O	"1,10-Phenanthroline;o-Phenanthroline ([M+Na]+)#Fenuron ([M+K39]+)#Kynuramine; 3-amino-1-(2-aminophenyl)propan-1-one [ketone] ([M+K39]+)"													null#null#null	null#null#null	null#null#null
217.07382	0.00E+00	216.0665435	7	217.073772498191#217.073772498191#217.073772498191#217.073772498191#217.073772498191#217.073772498191#217.073831814471	0.2188279038308#0.2188279038308#0.2188279038308#0.2188279038308#0.2188279038308#0.2188279038308#-0.0544260518389466	C01056#C03043#C16569#HMDB01240#HMDB01329#HMDB01497#C10963	C10H14N2O#C10H14N2O#C10H14N2O#C10H14N2O#C10H14N2O#C10H14N2O#C9H13ClN2O2	"(S)-6-Hydroxynicotine ([M+K39]+)#(R)-6-Hydroxynicotine ([M+K39]+)#Glycinexylidide;GX ([M+K39]+)#Pseudooxynicotine; 4-methylamino-1-pyridin-3-yl-butan-1-one [ketone] ([M+K39]+)#2'-Hydroxynicotine; 1-methyl-2-pyridin-3-yl-pyrrolidin-2-ol [hemiaminal] ([M+K39]+)#Nicotine-1'-N-oxide; 3-(1-methyl-1-oxidopyrrolidin-1-ium-2-yl)pyridine [N-oxide] ([M+K39]+)#Terbacil ([M+H]+)"													ko00760#ko00760#ko00982#null#null#null#null	";Nicotinate and nicotinamide metabolism#;Nicotinate and nicotinamide metabolism#;Drug metabolism - cytochrome P450#null#null#null#null"	null#null#null#null#null#null#null
219.0532	0.00E+00	180.0900406	5	219.053037053611#219.053037053611#219.053037053611#219.053037053611#219.053037053611	0.743866736468323#0.743866736468323#0.743866736468323#0.743866736468323#0.743866736468323	C05636#C05638#C12033#HMDB04076#HMDB13319	C9H12N2O2#C9H12N2O2#C9H12N2O2#C9H12N2O2#C9H12N2O2	"3-Hydroxykynurenamine ([M+K39]+)#5-Hydroxykynurenamine ([M+K39]+)#p-Aminophenylalanine ([M+K39]+)#5-Hydroxykynurenamine (see KEGG C05638); 3-amino-1-(2-amino-5-hydroxy-phenyl)propan-1-one [ketone] ([M+K39]+)#Tyrosinamide; 2-amino-3-(4-hydroxyphenyl)propanamide [phenol or hydroxyhetarene] ([M+K39]+)"													ko00380#ko00380#null#null#null	";Tryptophan metabolism#;Tryptophan metabolism#null#null#null"	null#null#null#null#null
221.06895	0.00E+00	198.0797293	3	221.068533650781#221.068533650781#221.068533650781	1.88334553086077#1.88334553086077#1.88334553086077	C06537#C14742#C15042	C12H10N2O#C12H10N2O#C12H10N2O	Harmalol ([M+Na]+)#N-Nitrosodiphenylamine ([M+Na]+)#4-(2-Pyrazinylethenyl)phenol ([M+Na]+)													null#null#null	null#null#null	null#null#null
235.08453	0.00E+00	196.1213706	3	235.084183714921#235.084183714921#235.084337181891	1.47302367786664#1.47302367786664#0.820207561142432	C01748#C06538#C13311	C13H12N2O#C13H12N2O#C10H16N2O2	"Pyocyanine;Reduced pyocyanine ([M+Na]+)#Harmine ([M+Na]+)#Fasoracetam;NS 105 ([M+K39]+)"													null#ko01063#null	"null#;Biosynthesis of alkaloids derived from shikimate pathway#null"	null#null#null
237.14837	0.00E+00	236.1410935	2	237.148520966961#237.148520966961	-0.636592868007426#-0.636592868007426	C14274#C14718	C14H20O3#C14H20O3	"4-Heptyloxybenzoic acid ([M+H]+)#Heptyl p-hydroxybenzoate;Heptylparaben ([M+H]+)"													null#null	null#null	null#null
243.21406	0.00E+00	242.2067835	1	243.2140984	-0.158000656	C11082	C15H30S	"2-N-Undecyltetrahydrothiophene;UTHT ([M+H]+)"													null	null	null
245.06914	0.00E+00	206.1059806	2	245.068687117751#245.068687117751	1.84797746869758#1.84797746869758	C07499#C12590	C11H14N2O2#C11H14N2O2	"Phenylethylmalonamide;2-Phenyl-2-ethylmalondiamide;PEMA ([M+K39]+)#Pheneturide;Ethylphenacemide ([M+K39]+)"													map07033#null	";Anticonvulsants#null"	null#null
245.19339	0.00E+00	244.1861135	1	245.193363	0.110184859	C05761	C14H27OSR	"R replaced by H in Tetradecanoyl-[acp];Tetradecanoyl-[acyl-carrier protein];Myristoyl-[acyl-carrier protein] ([M+H]+)"													"ko00061;ko00540;ko01100"	";Fatty acid biosynthesis;Lipopolysaccharide biosynthesis;Metabolic pathways"	X
249.06361	0.00E+00	210.1004506	4	249.063484620151#249.063484620151#249.063601737311#249.063601737311	0.503404929406317#0.503404929406317#0.0331750150970518#0.0331750150970518	C00647#HMDB01555#C07826#C15986	C8H13N2O5P#C8H13N2O5P#C10H14N2O3#C10H14N2O3	"Pyridoxamine phosphate;Pyridoxamine 5-phosphate;Pyridoxamine 5'-phosphate ([M+H]+)#Pyridoxamine 5'-phosphate (see KEGG C00647); [4-(aminomethyl)-5-hydroxy-6-methyl-pyridin-3-yl]methoxyphosphonic acid [phenol or hydroxyhetarene] ([M+H]+)#Aprobarbital ([M+K39]+)#2,6-Dihydroxypseudooxynicotine ([M+K39]+)"													"ko00750;ko01100#null#null#ko00760"	";Vitamin B6 metabolism;Metabolic pathways#null#null#;Nicotinate and nicotinamide metabolism"	X#null#null#null
249.18444	0.00E+00	248.1771635	13	249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681#249.184906475681	-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677#-1.87200966885677	LMFA01030163#LMFA01030164#LMFA01030165#LMFA01030166#LMFA01030277#LMFA01030278#LMFA01030279#LMFA01030280#LMFA01030491#LMFA01030492#LMFA01030493#LMFA01030870#LMFA01140010	C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2#C16H24O2	"16:4(4Z,7Z,10Z,13Z); 4,7,10,13-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#4,7,11,14-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#4,8,12,16-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#16:4(6Z,9Z,12Z,15Z); 6,9,12,15-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#2E,6Z,8Z,12E-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#2E,6Z,8Z,12Z-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#2Z,6Z,8Z,12E-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#2Z,6Z,8Z,12Z-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#3,9-hexadecadiynoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#7,10-hexadecadiynoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#8,10-hexadecadiynoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#16:4(6Z,9Z,12Z,15Z); 6Z,9Z,12Z,15Z-hexadecatetraenoic acid [Unsaturated fatty acids [FA0103]] ([M+H]+)#4-[3]-ladderane-butanoic acid [Carbocyclic fatty acids [FA0114]] ([M+H]+)"													null#null#null#null#null#null#null#null#null#null#null#null#null	null#null#null#null#null#null#null#null#null#null#null#null#null	null#null#null#null#null#null#null#null#null#null#null#null#null
raw_mass	peak_height	corrected_mass	npossible	KEGG_mass	ppm	KEGG_cid	KEGG_formula	KEGG_name	uniqueID	C13	O18	N15	S34	Mg25	Mg26	Fe54	Fe57	Ca44	Cl37	K41	KEGG Pathways	KEGG Pathways descriptions	Compound in Organism(X)"""

    def as_pandas(self):
        df = pd.read_csv(six.StringIO(self.as_str()), sep='\t', index_col=None, header=None)
        df.columns = list(df.iloc[-1])
        df = df.iloc[0:-1, :]
        return df.set_index(df.columns[0])

    def create_example_file(self, file_name):
        df = self.as_pandas()
        df.to_csv(file_name)


def demo_dataset(name):
    return DataSetFactory.create_dataset(name)


def demo(name):
    return DataSetFactory.create_dataset(name).as_pandas()


if __name__ == '__main__':
    from pandas.testing import assert_frame_equal
    import tempfile

    d = demo_dataset('demo1')
    print(d.as_str())
    print(d.as_pandas())
    print('NaNs:', d.as_pandas().isna().sum().sum())
    print('-'*40)
    print('testing demo1 roundtrip')
    with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
        tmp_file.close()
        d.create_example_file(tmp_file.name)
        print(tmp_file.name)
        d_back = pd.read_csv(tmp_file.name, index_col=0)
    assert_frame_equal(d.as_pandas(), d_back)
    print('round-trip ok!')
    print('-'*40)
    d = demo_dataset('demo2')

    print(d.as_str())
    print(d.as_pandas())
    print('-'*40)
    d = demo('demo2')
    print(d.columns.names)
    print(d.index.names)
    print('-'*40)

    print('testing demo2 roundtrip')
    d = demo_dataset('demo2')
    with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
        tmp_file.close()
        d.create_example_file(tmp_file.name)
        print(tmp_file.name)
        d_back = pd.read_csv(tmp_file.name, index_col=0, header=[0,1])
    assert_frame_equal(d.as_pandas(), d_back)
    print('round-trip ok!')
    print('-'*40)

    d = demo_dataset('table_with_formulae')
    print(d.as_str())
    print(d.as_pandas())
    print('-'*40)
    d = demo('table_with_formulae')
    print(d.columns.names)
    print(d.index.names)
    print('-'*40)
    d = demo_dataset('masstrix_output')
    print(d.as_str())
    print(d.as_pandas())
    print('-'*40)
    d = demo('masstrix_output')
    print(d.columns.names)
    print(d.index.names)
