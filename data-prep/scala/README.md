### Scripts for computing mutual information of pairs of labels

```scala

 val samplesFile = "/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/ontology/data/table_label.csv"
 
 //val samplesFile = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/samples/labels_samples.txt"
 
 // bucketization parameter
 val binNum = 10
 
 // loading samples: an array of table relevance scores for each label
 // and bucketizing relevance scores
 val samples = sc.textFile(samplesFile).map(line => line.split(",")).map(line => (line(0).toInt, line.drop(1).map(_.toDouble).map(x => math.floor(x*binNum).toInt)))
 
 val sampleNum = samples.count
 
 val tlps = samples.map{case (t,ps) => (t,ps.zipWithIndex)}.flatMap{case (t,pls) => pls.map{case (p,l) => (t,l,p)}}
 
 val pxs = sc.parallelize(tlps.map{case (t,l,p) => ((l,p),t)}.countByKey.toSeq).map{case ((l,p),f) => ((l,p),f/sampleNum.toDouble)}
 
 val xs = tlps.map{case (t,l,p) => (t,(l,p))}
 
 val pxys = sc.parallelize(xs.join(xs).filter{case (t,((l1,p1),(l2,p2))) => l1<l2}.map{case (t,((l1,p1),(l2,p2))) => ((l1, l2, p1, p2), t)}.countByKey.toSeq).map{case ((l1,l2,p1,p2),f) => (l1,l2,p1,p2,f/sampleNum.toDouble)}
 
 val mis = pxys.map{case (l1,l2,p1,p2,f12) => ((l1,p1),(l2,p2,f12))}.join(pxs).map{case ((l1,p1),((l2,p2,f12),f1)) => ((l2,p2),(l1,p1,f12,f1))}.join(pxs).map{case ((l2,p2), ((l1,p1,f12,f1),f2)) => ((l1,l2),f12*(math.log(f12/(f1*f2))/math.log(2)))}.reduceByKey(_ + _)

 ```
