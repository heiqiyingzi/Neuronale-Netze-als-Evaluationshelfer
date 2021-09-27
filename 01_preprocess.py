import questionnaire
import os

if __name__ == "__main__":

    path_boegen = 'Evaluationshelfer_Daten/boegen'
    ref_file = path_boegen + '/Bogen1.jpg'
    path_boxes_ref = 'Evaluationshelfer_Daten/boxes/Bogen1'
    if not os.path.exists(path_boxes_ref):
        os.makedirs(path_boxes_ref)
    ref_questionnaire= questionnaire.Questionnaire(ref_file, ref_boxes=None, ref_points=None, isReference=True, target_path=path_boxes_ref)
    print('Bogen1 Done')
    
    refBoxes = ref_questionnaire.ref_boxes
    refPoints = ref_questionnaire.ref_points


    for file in os.listdir(path_boegen):
        curr_file = path_boegen +'/' +  file
        if not curr_file==ref_file:
            path_boxes = 'Evaluationshelfer_Daten/boxes/'+ file[:-4]
            if not os.path.exists(path_boxes):
                os.makedirs(path_boxes)
            curr_questionnaire = questionnaire.Questionnaire(curr_file, ref_boxes=refBoxes, ref_points=refPoints, isReference=False, target_path=path_boxes)

        
            

        

  