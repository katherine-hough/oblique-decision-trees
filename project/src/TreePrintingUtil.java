import java.util.ArrayList;

/* Utility class that supports printing and visualizing DecisionTree instances */
public class TreePrintingUtil {

  /* Returns a string representation of the specified number of levels of the
   * specified decision tree */
  public static String getTreeString(DecisionTree root, int maxLevels) {
    ArrayList<ArrayList<DecisionTree>> levels = breadthFirstTraversal(root, maxLevels);
    int[] longest = getLongestStringPerLevel(levels);
    int numLines = 3*(levels.get(levels.size()-1).size()*2-1);
    String[] lines = new String[numLines];
    for(int i = 0; i < lines.length; i++) {
      lines[i] = "";
    }
    String[][][] levelLines = getLevelLines(levels, longest);
    for(int l = levels.size()-1; l>=0; l--) {
      int target = (int)Math.pow(2, levels.size()-1-l);
      int offset = 2*target;
      target--;
      int n = 0;
      int bandHeight = target;
      int bandTarget = (int)Math.pow(2, levels.size()-l);
      int bandOffset = 2*bandTarget;
      bandTarget--;
      for(int i = 0; i < lines.length; i+=3) {
        if((i/3) == target) {
          lines[i] += levelLines[l][n][0];
          lines[i+1] += levelLines[l][n][1];
          lines[i+2] += levelLines[l][n++][2];
          target += offset;
        } else {
          String end = " ";
          String end2 = " ";
          if(l != 0) {
            if(((i/3)>= bandTarget-bandHeight) && ((i/3)<= bandTarget+bandHeight)) {
              if(n > 0 && n < levels.get(l).size() && levels.get(l).get(n-1) != null && levels.get(l).get(n) != null) {
                end = "|";
                end2 = "|";
                if(i/3 == bandTarget) {
                  end2 = "+";
                }
              }
              if(i/3 == bandTarget+bandHeight) {
                bandTarget += bandOffset;
              }
            }
          }
          lines[i] += centerStringAtWidth("", longest[l], " ") + end;
          lines[i+1] += centerStringAtWidth("", longest[l], " ") + end2;
          lines[i+2] += centerStringAtWidth("", longest[l], " ") + end;
        }
      }
    }
    String result = "";
    for(String line : lines) {
      result += line + "\n";
    }
    return result;
  }

  /* Returns the lines for each node on each level */
  private static String[][][] getLevelLines(ArrayList<ArrayList<DecisionTree>> levels, int[] longest) {
    String[][][] levelLines = new String[levels.size()][][];
    for(int l = 0; l< levels.size(); l++) {
      ArrayList<DecisionTree> level = levels.get(l);
      levelLines[l] = new String[level.size()][3];
      for(int n = 0; n < level.size(); n++) {
        DecisionTree node = level.get(n);
        String cond = "";
        String freqs = "";
        String divide = " ";
        if(node!=null) {
          divide = "-";
          freqs = node.toString();
          if(node.getSplitCondition() != null) {
            cond = node.getSplitCondition().toString();
          }
        }
        String end1 = (l==0 || node == null) ? " " : "|";
        String end2 = (l==0 || node == null)? " " : "+";
        levelLines[l][n][0] = centerStringAtWidth(freqs, longest[l], " ") + (n%2==0 ? " " : end1);
        levelLines[l][n][1] = centerStringAtWidth("", longest[l], divide) + end2;
        levelLines[l][n][2] = centerStringAtWidth(cond, longest[l], " ") + (n%2==0 ? end1 : " ");
      }
    }
    return levelLines;
  }

  /* Returns the longest string length at each level of the tree. */
  private static int[] getLongestStringPerLevel(ArrayList<ArrayList<DecisionTree>> levels) {
    int[] longest = new int[levels.size()];
    for(int i = 0; i < levels.size(); i++) {
      for(DecisionTree tree : levels.get(i)) {
        if(tree != null) {
          longest[i] = Math.max(longest[i], tree.toString().length());
          if(tree.getSplitCondition() != null) {
            longest[i] = Math.max(longest[i], tree.getSplitCondition().toString().length());
          }
        }
      }
    }
    return longest;
  }

  /* Returns the specfied string but padded up to the specified minimum width with
   * the specified symbol and centered in that padding */
  public static String centerStringAtWidth(String s, int width, String symbol) {
    if(s.length() >= width) {
      return s; // no padding necessary
    }
    int padding = width - s.length();
    int frontPadding = padding/2;
    int backPadding = (padding+1)/2;
    String ret = "";
    for(int i = 0; i < frontPadding; i++) {
      ret += symbol;
    }
    ret +=s;
    for(int i = 0; i < backPadding; i++) {
      ret += symbol;
    }
    return ret;
  }

  /* Creates lists containing the nodes at each level of the tree for up to the
   * specified maximum number of levels. */
  private static ArrayList<ArrayList<DecisionTree>> breadthFirstTraversal(DecisionTree root, int maxLevels) {
    ArrayList<ArrayList<DecisionTree>> levels = new ArrayList<>();
    levels.add(new ArrayList<DecisionTree>());
    levels.get(0).add(root);
    int i = 0;
    while(levels.size() > i) {
      ArrayList<DecisionTree> nextLevel = new ArrayList<>();
      boolean nonNullChild = false;
      for(DecisionTree tree : levels.get(i)) {
        if(tree != null) {
          nextLevel.add(tree.getLeftChild());
          nextLevel.add(tree.getRightChild());
          if(tree.getLeftChild() != null || tree.getRightChild() != null) {
            nonNullChild = true;
          }
        } else {
          nextLevel.add(null);
          nextLevel.add(null);
        }
      }
      if(nonNullChild) {
        levels.add(nextLevel);
      }
      i++;
      if(i > maxLevels) {
        return levels;
      }
    }
    return levels;
  }
}
