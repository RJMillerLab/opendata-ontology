import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashMap

// calculating marginal probabilities
def mp(vNum: Int, vX: List[Int]) = {
    1.0 * vX.filter(_ == vNum).size / vX.size
}

// calculating joint probabilities
def jp(vNumA: Int, vA: List[Int], vNumB: Int, vB: List[Int]): Double = {
    val size = vA.size
    var count = 0
    for (i <- 0 until size) {
        if (vA(i) == vNumA && vB(i) == vNumB) {
            count += 1;
        }
    }
    count * 1.0 / size
}

// calculating mutual information
def mi(a: List[Int], b: List[Int]): Double = {
    val mpBuffer = HashMap[(String, Int), Double]()
    val jpBuffer = HashMap[(Int, Int), Double]()
    val ua = a.toSet.toList
    val ub = b.toSet.toList
    val r = for {
        x <- ua
        y <- ub
    } yield {
        val pxy = jpBuffer.getOrElseUpdate((x, y), jp(x, a, y, b))
        val px = mpBuffer.getOrElseUpdate(("x", x), mp(x, a))
        val py = mpBuffer.getOrElseUpdate(("y", y), mp(y, b))
        val r = pxy * (math.log(pxy / (px * py))/math.log(2))
        if (r.isInfinite() || r.isNaN()) 0 else r
    }
    r.sum
}

val labelsFile = "/home/fnargesian/go/src/github.com/RJMillerLab/opendata-ontology/data-prep/python/ontology/data/label_table.csv"

// bucketization parameter
val binNum = 10

// loading samples: an array of table relevance scores for each label
// and bucketizing relevance scores 
val labels = sc.textFile(labelsFile).map(line => line.split(",")).map(line => (line(0), line.drop(1).map(_.toDouble).map(x => math.floor(x*binNum).toInt)))

// generating pairs of labels
val pairs = labels.cartesian(labels).filter{case ((label1, tables1), (label2, tables2)) => label1 < label2}

// mutual information file
val misFile = "/home/fnargesian/FINDOPENDATA_DATASETS/10k/labels.mutual_info"

// computing mutual info for pairs
val mis = pairs.map{case((tid1, labels1), (tid2, labels2)) => ((tid1, tid2), mi(labels1.toList, labels2.toList))}.saveAsTextFile(misFile)
