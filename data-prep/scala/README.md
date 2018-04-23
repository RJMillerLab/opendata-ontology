### Scripts for computing mutual information for pairs of labels

```scala

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.HashMap
 
// calculating marginal probabilities
def mp(vNum: Int, vX: List[Int]) = {
     1.0 * vX.filter(_ == vNum).size / vX.size
}
```
